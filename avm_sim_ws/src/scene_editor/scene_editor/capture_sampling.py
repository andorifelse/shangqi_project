"""Geometry/depth preflight and observation-guided planar acquisition.

Camera truth is used only to render/select observations, never as solver input.
"""
from collections import Counter
from copy import deepcopy
import numpy as np
from scene_manager.camera_models import world_to_pixel,pixels_to_world_plane
from scene_manager.transforms import pose_matrix,matrix_pose,inverse,transform_points


class IntrinsicSampler:
    def __init__(self, scene, name, board, request, rng, background_depth=None):
        self.scene,self.name,self.board,self.request,self.rng=scene,name,board,request,rng
        self.depth=background_depth
        self.plane=pose_matrix(board.pose)
        self.origin=scene.camera_optical_pose(name)[:3,3]
        rows,cols=request['pattern_size'][1],request['pattern_size'][0]
        corners=sorted(request['intrinsic_geometry']['corners'],key=lambda row:(row['grid_row'],row['grid_col']))
        self.grid=np.array([r['xyz_m'] for r in corners]).reshape(rows,cols,3)
        dx=self.grid[0,1]-self.grid[0,0];dy=self.grid[1,0]-self.grid[0,0]
        margin=request['intrinsic_geometry'].get('white_margin_m',0.)
        dx=dx*(1+margin/np.linalg.norm(dx));dy=dy*(1+margin/np.linalg.norm(dy))
        self.outer=np.array([self.grid[0,0]-dx-dy,self.grid[0,-1]+dx-dy,
            self.grid[-1,-1]+dx+dy,self.grid[-1,0]-dx+dy])
        self.anchors=[];self.failed=[];self.used=[];self.rejections=Counter()
        self.radius=1.;self.first=True
        self.centres=None
        if request['sampling']=='visible_plane':
            camera=scene.cameras[name]
            u,v=np.meshgrid(np.linspace(.05,.95,25)*camera.width,np.linspace(.05,.95,19)*camera.height)
            normal=self.plane[:3,2]
            hits,ok=pixels_to_world_plane(scene,name,np.stack([u,v],axis=-1).reshape(-1,2),
                normal,-float(normal@self.plane[:3,3]))
            self.centres=transform_points(inverse(self.plane),hits[ok])
        self.moving=any(request[k]>0 for k in ('spread_x','spread_y','yaw_range_deg'))

    def evaluate(self, pose):
        camera=self.scene.cameras[self.name];t=pose_matrix(pose)
        points=transform_points(t,self.grid.reshape(-1,3))
        uv,valid=world_to_pixel(self.scene,self.name,points)
        outer,outer_valid=world_to_pixel(self.scene,self.name,transform_points(t,self.outer))
        if not (valid.all() and outer_valid.all() and np.all(outer>=[4,4])
                and np.all(outer<[camera.width-5,camera.height-5])):
            return None,'outside_camera_domain'
        grid=uv.reshape(*self.grid.shape[:2],2)
        dx=np.diff(grid,axis=1);dy=np.diff(grid,axis=0)
        spacing=min(np.linalg.norm(dx,axis=-1).min(),np.linalg.norm(dy,axis=-1).min())
        if spacing<8.:return None,'projected_squares_too_small'
        a=dx[0,0];b=dy[0,0]
        sine=abs(a[0]*b[1]-a[1]*b[0])/(np.linalg.norm(a)*np.linalg.norm(b))
        if sine<.25:return None,'board_too_oblique'
        if self.depth is not None:
            pixels=np.rint(uv).astype(int)
            depths=self.depth[pixels[:,1],pixels[:,0]]
            distances=np.linalg.norm(points-self.origin,axis=1)
            occluded=np.isfinite(depths)&(distances>depths+np.maximum(.02,.02*depths))
            if np.mean(occluded)>.1:return None,'occluded_by_static_scene'
        centre=uv.mean(axis=0)
        angle=np.arctan2((inverse(self.plane)@t)[1,0],(inverse(self.plane)@t)[0,0])
        def close(previous,pixels,degrees):
            turn=abs(np.arctan2(np.sin(angle-previous['angle']),np.cos(angle-previous['angle'])))
            return np.linalg.norm(centre-previous['centre'])<pixels and turn<np.deg2rad(degrees)
        if self.moving and any(close(p,4.,2.) for p in self.used):return None,'duplicate_observation'
        if any(close(p,12.,5.) for p in self.failed):return None,'near_failed_observation'
        score=min(spacing,40.)*np.sqrt(sine)
        if self.used:
            separation=min(np.linalg.norm(centre-p['centre']) for p in self.used)
            score*=.5+.5*min(separation/40.,1.)
        return {'pose':pose,'uv':uv,'outer_uv':outer,'centre':centre,'angle':angle,
            'score':float(score),'min_square_px':float(spacing)},None

    def propose(self):
        from .scene_pipeline import sample_intrinsic_pose
        if self.first:
            self.first=False
            candidate,reason=self.evaluate(list(self.board.pose))
            if candidate is not None:return candidate
            self.rejections[reason]+=1
        candidates=[]
        # Scan cheap proposals before committing to any expensive full-scene render.
        for i in range(64 if self.anchors else 256):
            if self.anchors and (i%4!=3 or self.request['sampling']=='around_pose'):
                anchor=self.anchors[int(self.rng.integers(len(self.anchors)))]
                local=inverse(self.plane)@pose_matrix(anchor['pose'])
                dx=local[0,3]+self.rng.uniform(-1.,1.)*self.request['spread_x']*self.radius
                dy=local[1,3]+self.rng.uniform(-1.,1.)*self.request['spread_y']*self.radius
                if self.request['sampling']=='around_pose':
                    dx=np.clip(dx,-self.request['spread_x'],self.request['spread_x'])
                    dy=np.clip(dy,-self.request['spread_y'],self.request['spread_y'])
                limit=np.deg2rad(self.request['yaw_range_deg'])
                angle=np.clip(anchor['angle']+self.rng.uniform(-limit,limit)*self.radius,-limit,limit)
                pose=matrix_pose(self.plane@pose_matrix([dx,dy,0.,0.,0.,angle]))
            elif self.centres is not None:
                if not len(self.centres):
                    self.rejections['ray_does_not_intersect_forward_plane']+=1
                    continue
                centre=self.centres[int(self.rng.integers(len(self.centres)))]
                angle=np.deg2rad(self.rng.uniform(-self.request['yaw_range_deg'],self.request['yaw_range_deg']))
                pose=matrix_pose(self.plane@pose_matrix([*centre[:2],0.,0.,0.,angle]))
            else:
                pose=sample_intrinsic_pose(self.scene,self.name,self.board,self.request,self.rng)
            if pose is None:
                self.rejections['ray_does_not_intersect_forward_plane']+=1
                continue
            candidate,reason=self.evaluate(pose)
            if candidate is None:self.rejections[reason]+=1
            else:candidates.append(candidate)
        if not candidates:return None
        return max(candidates,key=lambda row:row['score'])

    def observe(self, candidate, found):
        if found:
            self.used.append(candidate);self.anchors.append(candidate)
            self.anchors=self.anchors[-16:];self.radius=min(1.,self.radius*1.15)
        else:
            self.failed.append(candidate);self.failed=self.failed[-64:]
            self.radius=max(.2,self.radius*.7)


def plan_intrinsic_board(scene, name, board, request, seed, background_depth=None):
    """Select one fixed plane per independent camera; never alter camera poses."""
    def check(candidate_board):
        sampler=IntrinsicSampler(scene,name,candidate_board,request,np.random.default_rng(seed),background_depth)
        candidate=sampler.propose()
        return sampler,candidate
    original,candidate=check(board)
    if request['sampling']=='around_pose' or (candidate is not None and candidate['min_square_px']>=16.):
        return deepcopy(board),'scene_plane',candidate,dict(original.rejections)
    # The 1.0.0 solver fits each camera independently, so different cameras may
    # use different planes. A single camera's sequence remains strictly coplanar.
    camera=scene.cameras[name];optical=scene.camera_optical_pose(name)
    rotation=optical[:3,:3]@np.diag([1.,-1.,-1.])
    centre=original.grid.reshape(-1,3).mean(axis=0)
    distance=float(np.clip(min(camera.K[0],camera.K[4])*request['square_size_m']/32.,.6,3.))
    for factor in (1.,.75,.5,1.25):
        transform=np.eye(4);transform[:3,:3]=rotation
        transform[:3,3]=optical[:3,3]+optical[:3,2]*distance*factor-rotation@centre
        # Keep the full plate above the configured ground, without changing the
        # geometry, dimensions or any other object in the scene.
        bottom=transform_points(transform,original.outer)[:,2].min()
        transform[2,3]+=max(0.,scene.coordinate.ground_z+.03-bottom)
        planned=deepcopy(board);planned.pose=matrix_pose(transform)
        sampler,proposal=check(planned)
        if proposal is not None and proposal['min_square_px']>=16.:
            return planned,'camera_facing_plane',proposal,dict(sampler.rejections)
    return deepcopy(board),'scene_plane',candidate,dict(original.rejections)
