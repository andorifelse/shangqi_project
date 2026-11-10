"""Current SceneState -> direct lens PNG -> unmodified pipeline 1.0.0 solvers.

Capture and evaluation may read truth. Generated solver specs may NOT. This is
an explicit alternate acquisition adapter, not Graphdeco cubemap emulation.
"""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from scene_manager.board_geometry import board_geometry
from scene_manager.transforms import pose_matrix,matrix_pose,inverse,transform_points


def put(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def region_grids(document):
    groups={}
    for row in document['corners']:groups.setdefault(row['region'],[]).append(row)
    result={}
    for name,items in groups.items():
        rs=sorted({x['grid_row'] for x in items});cs=sorted({x['grid_col'] for x in items})
        if len(rs)<3 or len(cs)<3 or len(items)!=len(rs)*len(cs):
            raise ValueError(f'区域 {name}：需要至少 3×3 的完整连续矩形角点子区域，不可跨中央孔')
        if rs!=list(range(rs[0],rs[-1]+1)) or cs!=list(range(cs[0],cs[-1]+1)):
            raise ValueError(f'区域 {name}：行列编号必须连续')
        lookup={(x['grid_row'],x['grid_col']):x for x in items}
        grid=np.array([[lookup[r,c]['xyz_m'] for c in cs] for r in rs])
        dx=np.diff(grid,axis=1);dy=np.diff(grid,axis=0)
        pitch=float(np.linalg.norm(dx[0,0]));vertical=np.linalg.norm(dy,axis=-1)
        if (pitch<=0 or not np.allclose(dx,dx[0,0],atol=1e-6,rtol=1e-5)
                or not np.allclose(dy,dy[0,0],atol=1e-6,rtol=1e-5)
                or not np.allclose(vertical,pitch,atol=1e-6,rtol=1e-5)
                or not np.isclose(dx[0,0]@dy[0,0],0,atol=1e-7)):
            raise ValueError(f'区域 {name}：角点不是等距正方网格；请核实真实格长和方向')
        if not np.allclose(grid[:,:,2],grid[0,0,2],atol=1e-7):
            raise ValueError(f'区域 {name}：1.0.0 要求水平板，请修正板/资产局部对齐')
        result[name]={'rows':len(rs),'columns':len(cs),'pitch':pitch,'grid':grid,
            'items':[lookup[r,c] for r in rs for c in cs]}
    return result


def normalize_scene_request(scene, request):
    """Cheap schema/geometry checks only; no PLY reading or CUDA on UI thread."""
    out=deepcopy(request);names=out.get('cameras') or sorted(scene.cameras)
    if not names or set(names)-scene.cameras.keys():raise ValueError('请选择当前场景中存在的相机')
    if len(set(names))!=len(names):raise ValueError('相机不能重复选择')
    for name in names:
        if scene.cameras[name].model not in ('pinhole','fisheye'):
            raise ValueError(f'{name}：1.0.0 仅支持 pinhole/fisheye，不能将 ftheta 冒充 fisheye')
    small_id=out.get('intrinsic_board');large_id=out.get('extrinsic_board')
    if small_id not in scene.boards:raise ValueError('请选择内参采集小板')
    if large_id not in scene.boards:raise ValueError('请选择外参大板（可以是环车多 region 高斯板）')
    if small_id==large_id:raise ValueError('内参运动小板和外参固定大板必须是两个不同对象')
    small=scene.boards[small_id];large=scene.boards[large_id]
    small_doc=board_geometry(small);large_doc=board_geometry(large)
    grids=region_grids(small_doc)
    if len(grids)!=1:raise ValueError('内参小板必须是单一完整矩形区域，环车条带仅作为外参板')
    grid=next(iter(grids.values()))
    if not np.allclose(grid['grid'][:,:,2],0.,atol=1e-7):
        raise ValueError('内参板公开角点必须在 board 局部 Z=0 平面；请先校正 asset_pose/角点原点')
    world_doc=deepcopy(large_doc)
    for row in world_doc['corners']:
        row['xyz_m']=transform_points(pose_matrix(large.pose),row['xyz_m']).tolist()
    world_grids=region_grids(world_doc)
    mapping=out.get('camera_regions') or scene.calibration.get('camera_regions',{})
    missing=[n for n in names if mapping.get(n) not in world_grids]
    if missing:raise ValueError('请明确设置 camera→region：'+', '.join(missing)+'；不会根据相机名字猜区域')
    for key,lo,hi,default in (('views',4,100,12),('holdout',1,30,2),('seed',0,2**32-1,20261020),
                             ('supersample',1,4,1),('max_attempts',1,100,20)):
        value=out.get(key,default)
        if isinstance(value,bool) or not isinstance(value,int) or not lo<=value<=hi:
            raise ValueError(f'{key} 必须为 {lo}–{hi} 的整数')
        out[key]=value
    for key,default,limit in (('spread_x',.5,20.),('spread_y',.5,20.),('yaw_range_deg',30.,180.)):
        value=out.get(key,default)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<=limit:
            raise ValueError(f'{key} 必须为 0–{limit} 的有限数值')
        out[key]=float(value)
    if out.get('backend','gsplat') not in ('gsplat','cpu'):raise ValueError('渲染后端只能选择 gsplat 或 cpu')
    if out.get('sampling','visible_plane') not in ('visible_plane','around_pose'):
        raise ValueError('小板采样方式必须为视野内板平面或当前板附近')
    b=scene.bev
    mx=(b.x_max-b.x_min)/b.height;my=(b.y_max-b.y_min)/b.width
    if not np.isclose(mx,my):raise ValueError('1.0.0 BEV 仅支持正方形像素；请使 X范围/高度 = Y范围/宽度')
    out.update(cameras=names,camera_regions={n:mapping[n] for n in names},backend=out.get('backend','gsplat'),
        intrinsic_geometry=small_doc,extrinsic_geometry=large_doc,
        pattern_size=[grid['columns'],grid['rows']],square_size_m=grid['pitch'],
        sampling=out.get('sampling','visible_plane'))
    return out


def evaluation_frame(scene):
    """BEV's vehicle-aligned, level frame; do NOT pass this camera truth to solver."""
    if scene.vehicle:
        p=scene.vehicle.pose
        return pose_matrix([p[0],p[1],scene.coordinate.ground_z,0.,0.,p[5]])
    return pose_matrix([0.,0.,scene.coordinate.ground_z,0.,0.,0.])


def public_corners(scene, request):
    board=scene.boards[request['extrinsic_board']]
    t=inverse(evaluation_frame(scene))@pose_matrix(board.pose)
    corners=[]
    for row in request['extrinsic_geometry']['corners']:
        corners.append({k:row[k] for k in ('id','region','grid_row','grid_col')})
        corners[-1]['xyz_world_m']=transform_points(t,row['xyz_m']).tolist()
    return {'units':'m','coordinate_frame':'world','frame_definition':'vehicle-level BEV frame; see coordinate_contract.json',
        'source':request['extrinsic_geometry']['source'],'corners':corners,
        'camera_regions':request['camera_regions']}


def sample_intrinsic_pose(scene, name, board, request, rng):
    """Move within the initial board plane, regardless of its world tilt."""
    from scene_manager.camera_models import pixels_to_world_plane
    plane=pose_matrix(board.pose)
    if request['sampling']=='visible_plane':
        camera=scene.cameras[name]
        pixel=np.array([rng.uniform(.25,.75)*camera.width,rng.uniform(.3,.85)*camera.height])
        normal=plane[:3,2]
        hit,ok=pixels_to_world_plane(scene,name,pixel,normal,-float(normal@plane[:3,3]))
        if not bool(ok):return None
        local=transform_points(inverse(plane),hit)
        dx,dy=local[:2]
    else:
        dx=rng.uniform(-request['spread_x'],request['spread_x'])
        dy=rng.uniform(-request['spread_y'],request['spread_y'])
    yaw=np.deg2rad(rng.uniform(-request['yaw_range_deg'],request['yaw_range_deg']))
    # Right multiplication keeps both translation and rotation in board coordinates.
    return matrix_pose(plane@pose_matrix([dx,dy,0.,0.,0.,yaw]))


def capture_scene(scene, request, directory, progress):
    """Generate real PNG observations. No analytic replacement for Gaussian assets."""
    from gs_renderer.pipeline import SensorPipeline
    from .capture_sampling import IntrinsicSampler,plan_intrinsic_board
    from scene_manager.camera_models import camera_rays
    directory=Path(directory);small_id=request['intrinsic_board']
    small=scene.boards[small_id]
    pipeline=SensorPipeline(request['backend']);records=[];splits={};poses=[];attempt_log=[]
    timing={'cameras':{},'render_s':0.,'detection_s':0.,'prefilter_s':0.,'background_s':0.}
    started=time.perf_counter()
    root=directory/'pipeline';capture_root=directory/'capture';capture_root.mkdir()
    # All selected cameras stay fixed, with current K/D/resolution. A camera's
    # protocol is independent and each camera uses one fixed physical plane.
    fixed={n:scene.camera_optical_pose(n).tolist() for n in request['cameras']}
    render_truth={'cameras':{n:{'definition':scene.cameras[n].__dict__,
        'T_world_from_camera_optical':fixed[n]} for n in fixed},
        'scene_revision':request.get('scene_revision'), 'backend':request['backend'],
        'environment':{'scale':scene.coordinate.scale,'T_world_from_gs':scene.coordinate.T_world_from_gs},
        'vehicle':None if not scene.vehicle else {'pose':scene.vehicle.pose,'asset_pose':scene.vehicle.asset_pose,
            'asset_scale':scene.vehicle.asset_scale},
        'boards':{n:{'pose':b.pose,'asset_pose':b.asset_pose,'asset_scale':b.asset_scale} for n,b in scene.boards.items()}}
    put(capture_root/'render_truth.json',render_truth)
    contract={'adapter':'direct_lens_png_v1','renderer':request['backend'],
        'not_graphdeco_cubemap':True,'skipped_original_stages':['prepare','capture','compose'],
        'reason':'1.0.0 accepts observed PNG; direct current calibrated lenses preserve arbitrary K/D and resolution',
        'protocol':'fixed_camera_plane_xy_yaw',
        'small_plane_origin_world_m':small.pose[:3],
        'small_plane_normal_world':pose_matrix(small.pose)[:3,2].tolist(),
        'pose_sequence':'per-camera seeded independent planar observations, not simultaneous joint-BA observations',
        'target_isolation':'intrinsics preserves all boards; after all intrinsic captures only the intrinsic board is removed for extrinsics/BEV; other boards/environment/vehicle preserved',
        'small_motion':'per-camera fixed-plane local X/Y translations and normal-axis rotation; auto mode may plan a camera-facing plane before capture',
        'sampling':request['sampling'],
        'sampling_acceleration':'geometry/depth preflight; successful-pose neighbourhoods with global exploration',
        'pixel':'integer OpenCV centers; RGB uint8 -> cv2 BGR encode PNG, no solver resize',
        'gsplat_K':'internal rasterizer-only principal point +0.5; saved K is unchanged',
        'asset_to_world':'T_world_board @ T_board_asset @ (asset_scale * p_asset)',
        'environment':'T_world_from_gs @ (coordinate.scale * p_gs)',
        'camera':'T_world_base @ T_base_camera_link @ T_link_optical (or world parent)',
        'T_world_from_evaluation':evaluation_frame(scene).tolist(),
        'evaluation_frame':'level vehicle yaw/XY at ground_z; public corners and estimated extrinsics use this frame',
        'difference_from_live_AVM':'pipeline evaluation frame is explicitly level; for rolled/pitched vehicles it differs from live base XY ground intersection',
        'solver_inputs':'only PNG, public grid sizes/splits, public board corners and camera-region associations',
        'gsplat_limitations':'production DC-only colors; no higher SH; not pixel-equivalent to Graphdeco cubemap',
        'GLB_vehicle':'editor-only; no Gaussian occlusion for GLB/GLTF assets',
        'detection':'real PNG SB detections only; no projected GT pixel substitution or ID resolution'}
    put(directory/'coordinate_contract.json',contract)
    total=(request['views']+request['holdout']+1)*len(fixed);done=0
    pose_path=capture_root/'board_poses.csv'
    with pose_path.open('w',newline='',encoding='utf-8') as f:
        csv.writer(f).writerow(['camera','image','split','board','x','y','z','roll','pitch','yaw'])
    def render(sample,name):
        camera=deepcopy(sample.cameras[name]);w,h=camera.width,camera.height;factor=request['supersample']
        camera.width*=factor;camera.height*=factor
        for i in (0,4):camera.K[i]*=factor
        for i in (2,5):camera.K[i]=(camera.K[i]+.5)*factor-.5
        sample.cameras={name:camera}
        if not np.allclose(sample.camera_optical_pose(name),fixed[name],atol=1e-10):
            raise RuntimeError('采集相机位姿意外发生变化')
        image=pipeline.render(sample)[name].rgb
        if factor>1:image=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
        return image
    def save_image(image,name,kind,filename):
        c=scene.cameras[name];path=root/'images'/kind/c.model/name/filename
        path.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR)):raise OSError(path)
        records.append({'camera':name,'kind':kind,'model':c.model,'file':str(path.relative_to(directory)),
            'sha256':digest(path),'resolution_px':[c.width,c.height]})
        put(capture_root/'manifest.json',{'adapter':'direct_lens_png_v1','records':records,'complete':False})
    put(capture_root/'manifest.json',{'adapter':'direct_lens_png_v1','records':[],'complete':False})
    put(capture_root/'sampling_attempts.json',[])
    # Check every camera before spending minutes collecting the first camera.
    preflight={};planned_boards={}
    def plane_record(board,source,candidate,rejections):
        return {'feasible':candidate is not None,'source':source,'pose':board.pose,
            'normal_world':pose_matrix(board.pose)[:3,2].tolist(),
            'min_square_px':None if candidate is None else candidate['min_square_px'],
            'rejections':rejections}
    for name in request['cameras']:
        seed=int.from_bytes(hashlib.sha256(f"{request['seed']}:{name}".encode()).digest()[:8],'little')
        planned,source,candidate,rejections=plan_intrinsic_board(scene,name,small,request,seed)
        planned_boards[name]=planned
        preflight[name]=plane_record(planned,source,candidate,rejections)
    put(capture_root/'preflight.json',preflight)
    missing=[name for name,row in preflight.items() if not row['feasible']]
    if missing:
        for name in missing:
            attempt_log.append({'camera':name,'attempt':1,'pose':None,'reason':'no_feasible_board_pose',
                'prefilter_rejections':preflight[name]['rejections']})
        put(capture_root/'sampling_attempts.json',attempt_log)
        raise RuntimeError('、'.join(missing)+f"：仅采到 0/{request['views']+request['holdout']} 张；几何预检未找到完整可见且格子足够大的内参板姿态。请调整板位置或采样模式；自动模式支持逐相机布板，当前小板附近模式保留原平面，相机始终固定。")
    for name in request['cameras']:
        c=scene.cameras[name];seed=int.from_bytes(hashlib.sha256(f"{request['seed']}:{name}".encode()).digest()[:8],'little')
        mask_path=capture_root/'valid'/f'{name}.png';mask_path.parent.mkdir(exist_ok=True)
        _,valid_rays=camera_rays(c)
        if not cv2.imwrite(str(mask_path),valid_rays.astype(np.uint8)*255):raise OSError(mask_path)
        rng=np.random.default_rng(seed);accepted=[];attempts=0;need=request['views']+request['holdout']
        progress(f'准备 {name} 静态场景遮挡缓存',done=done,total=total)
        background=deepcopy(scene);background.cameras={name:background.cameras[name]}
        del background.boards[small_id]
        t0=time.perf_counter()
        # Original-resolution depth suffices for conservative candidate screening.
        background_result=pipeline.render(background)[name]
        depth=getattr(background_result,'depth',None)
        timing['background_s']+=time.perf_counter()-t0
        # Recheck against actual static depth before committing to the plane.
        planned,source,candidate,rejections=plan_intrinsic_board(scene,name,planned_boards[name],request,seed,depth)
        if source=='scene_plane':source=preflight[name]['source']
        planned_boards[name]=planned
        preflight[name]=plane_record(planned,source,candidate,rejections)
        put(capture_root/'preflight.json',preflight)
        contract['intrinsic_planes_world']=preflight
        render_truth['intrinsic_planes_world']=preflight
        put(directory/'coordinate_contract.json',contract)
        put(capture_root/'render_truth.json',render_truth)
        if source=='camera_facing_plane':
            progress(f'{name} 自动布置内参板，固定该路采集平面',done=done,total=total)
        sampler=IntrinsicSampler(scene,name,planned,request,rng,depth)
        del background_result
        while len(accepted)<need and attempts<need*request['max_attempts']:
            t0=time.perf_counter();candidate=sampler.propose()
            prefilter_s=time.perf_counter()-t0;timing['prefilter_s']+=prefilter_s
            if candidate is None:
                attempt_log.append({'camera':name,'attempt':attempts+1,'pose':None,
                    'reason':'no_feasible_board_pose','prefilter_rejections':dict(sampler.rejections)})
                put(capture_root/'sampling_attempts.json',attempt_log)
                break
            attempts+=1;pose=candidate['pose']
            sample=deepcopy(scene);sample.cameras={name:sample.cameras[name]}
            sample.boards[small_id].pose=pose
            progress(f'采集内参 {name} {len(accepted)}/{need} · 尝试 {attempts} · 渲染',done=done,total=total)
            t0=time.perf_counter();image=render(sample,name);render_s=time.perf_counter()-t0
            timing['render_s']+=render_s
            progress(f'采集内参 {name} {len(accepted)}/{need} · 尝试 {attempts} · 检测角点',done=done,total=total)
            t0=time.perf_counter()
            found,_=cv2.findChessboardCornersSB(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),tuple(request['pattern_size']),
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY)
            detection_s=time.perf_counter()-t0;timing['detection_s']+=detection_s
            sampler.observe(candidate,found)
            if found:
                index=len(accepted);filename=f'view_{index:04d}.png'
                split='train' if index<request['views'] else 'holdout'
                save_image(image,name,'intrinsics',filename);accepted.append({'image':filename,'split':split})
                poses.append({'camera':name,'image':filename,'split':split,'board':small_id,'pose':pose})
                with pose_path.open('a',newline='',encoding='utf-8') as f:
                    csv.writer(f).writerow([name,filename,split,small_id,*pose])
                done+=1;progress(f'采集内参 {name} {index+1}/{need}',done=done,total=total)
                reason='accepted'
            else:
                reason='SB_detection_failed_or_occluded'
                rejected=capture_root/'rejected'/name/f'attempt_{attempts:04d}.png';rejected.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(rejected),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            attempt_log.append({'camera':name,'attempt':attempts,'pose':pose,'reason':reason,
                'render_s':render_s,'detection_s':detection_s,'prefilter_s':prefilter_s,
                'min_square_px':candidate['min_square_px']})
            timing['cameras'][name]={'rendered_attempts':attempts,'accepted':len(accepted),
                'prefilter_rejections':dict(sampler.rejections)}
            timing['elapsed_s']=time.perf_counter()-started
            put(capture_root/'timing.json',timing)
            put(capture_root/'sampling_attempts.json',attempt_log)
            put(capture_root/'manifest.json',{'adapter':'direct_lens_png_v1','records':records,'complete':False})
        if len(accepted)<need:
            put(capture_root/'manifest.json',{'adapter':'direct_lens_png_v1','records':records,'complete':False})
            raise RuntimeError(f'{name}：仅采到 {len(accepted)}/{need} 张可检测图片。相机固定且小板仅在该相机的固定板平面内平移/旋转；请将小板放入该相机视野并检查遮挡、板平面或采样范围。拒绝图像和尝试已保存在 capture/。')
        splits[name]=accepted
    # Finish every camera's intrinsic capture before removing the moving target.
    # Work on the experiment copy so the saved/UI scene remains reusable.
    subsequent=deepcopy(scene)
    del subsequent.boards[small_id]
    progress(f'内参图片采集完成，后续阶段移除内参板 {small_id}，其他标定板保持不变',done=done,total=total)
    for name in request['cameras']:
        sample=deepcopy(subsequent);sample.cameras={name:sample.cameras[name]}
        image=render(sample,name);save_image(image,name,'extrinsics','big_board_fixed.png')
        done+=1;progress(f'采集外参 {name}',done=done,total=total)
    manifest={'adapter':'direct_lens_png_v1','backend':request['backend'],'complete':True,'records':records,
        'fixed_camera_truth':'render_truth.json (capture/evaluation only)','camera_motion':False,
        'intrinsic_excluded_board':None,'extrinsic_excluded_board':small_id,
        'intrinsic_boards':list(scene.boards),'extrinsic_boards':list(subsequent.boards),
        'intrinsic_planes_world':preflight,
        'small_plane_z_m':small.pose[2],
        'small_plane_normal_world':pose_matrix(small.pose)[:3,2].tolist(), 'seed':request['seed']}
    asset_file=directory/'logs/assets.json'
    if asset_file.exists():
        for info in json.loads(asset_file.read_text()).values():
            if digest(info['path'])!=info['sha256']:
                raise RuntimeError('采集期间资产文件发生变化，拒绝标记实验完成：'+info['path'])
        manifest['assets_unchanged']=True
    timing['elapsed_s']=time.perf_counter()-started
    put(capture_root/'timing.json',timing)
    put(capture_root/'manifest.json',manifest)
    return splits


def configs(scene, request, directory, splits):
    """Return ONLY publicly allowed solver specifications, with per-size groups."""
    root=Path(directory)/'pipeline';inputs=Path(directory)/'solver_inputs';inputs.mkdir()
    public=public_corners(scene,request);put(inputs/'public_corners.json',public)
    groups={}
    for name in request['cameras']:
        c=scene.cameras[name];groups.setdefault((c.model,c.width,c.height),[]).append(name)
    result=[]
    for i,((model,w,h),names) in enumerate(groups.items()):
        solver={'geometry':{'pattern_size':request['pattern_size'],'square_size':request['square_size_m']},
            'camera_ids':names,'models':{model:{'resolution_px':[w,h]}},'seeds':[0,1,2],
            'acquisition_protocol':'fixed_camera_plane_xy_yaw',
            'views':{model:{n:splits[n] for n in names}},'minimum_training_images':4}
        # Test this allowlist recursively; NEVER add K/D/pose/sample metadata.
        path=inputs/f'solver_group_{i}.json';put(path,solver)
        spec={'schema_version':'calibration-pipeline.v1','output_dir':str(root/'groups'/str(i)),
            'stages':['intrinsics','extrinsics'],
            'intrinsics':{'images':str(root/'images/intrinsics'),'solver_spec':str(path),'channels':names},
            'extrinsics':{'images':str(root/'images/extrinsics'),'corners_world':str(inputs/'public_corners.json'),
                'regions':{n:request['camera_regions'][n] for n in names},'models':[model]}}
        put(inputs/f'pipeline_group_{i}.json',spec);result.append(spec)
    return result


def evaluate(scene, request, directory):
    """Post-fit comparison; never returns truth to the estimator or candidate selector."""
    from calibration_pipeline.bev import select_candidate
    root=Path(directory)/'pipeline';rows=[]
    for name in request['cameras']:
        c=scene.cameras[name];ipath=root/'intrinsics'/c.model/name/'calibration.json'
        if not ipath.exists():continue
        intr=json.loads(ipath.read_text());truth=inverse(evaluation_frame(scene))@scene.camera_optical_pose(name)
        solutions=[]
        for s in intr.get('solutions',[]):
            k=np.array(s['K']);d=np.array(s['D'])
            solutions.append({'seed':s['seed'],'K_est_minus_truth':(k-c.matrix).tolist(),
                'D_est_minus_truth':(d-np.array(c.D)).tolist(),
                'focal_relative_percent':float(100*np.linalg.norm([k[0,0]-c.K[0],k[1,1]-c.K[4]])/np.linalg.norm([c.K[0],c.K[4]]))})
        row={'camera':name,'model':c.model,'intrinsic_status':intr['status'],'intrinsic_candidates':solutions,
            'not_acceptance':True}
        epath=root/'extrinsics'/c.model/name/'extrinsics.json'
        if epath.exists():
            ext=json.loads(epath.read_text());candidates=[]
            for i,s in enumerate(ext.get('candidates',[])):
                r=np.array(s['R_w2c']);t=np.array(s['tvec']);center=-r.T@t
                angle=float(np.rad2deg(np.arccos(np.clip((np.trace(r.T@truth[:3,:3].T)-1)/2,-1,1))))
                candidates.append({'index':i,'intrinsic_seed':s['intrinsic_seed'],
                    'translation_error_m':float(np.linalg.norm(center-truth[:3,3])),
                    'rotation_error_deg':angle,'all_correspondence_rmse_px':s['all_correspondence_rmse_px']})
            row.update(extrinsic_status=ext['status'],extrinsic_candidates=candidates)
            if candidates:
                # Use original selector based ONLY on observed reprojection, never lowest GT error.
                try:
                    index,_,_,_=select_candidate(ext,intr,c.model)
                    row['selected_by_observations']=candidates[index]
                except ValueError as exc:row['selection_error']=str(exc)
        rows.append(row)
    put(Path(directory)/'ground_truth_comparison.json',{'usage':'evaluation only, after solver execution',
        'frame':'vehicle-level BEV frame','results':rows})
    return rows


def run_scene_pipeline(scene, request, directory, Pipeline, progress):
    directory=Path(directory);root=directory/'pipeline'
    request=normalize_scene_request(scene,request)
    progress('检查资产与直出 PNG 采集能力')
    if request['backend']=='gsplat':
        import torch
        if not torch.cuda.is_available():raise RuntimeError('gsplat 无法访问 CUDA；请检查驱动，或仅在小场景选择 CPU')
    splits=capture_scene(scene,request,directory,progress)
    specs=configs(scene,request,directory,splits);runs=[];failures=[]
    put(directory/'generated_config.json',{'adapter':'direct_lens_png_v1','revision':request.get('scene_revision'),
        'roles':{'intrinsics':request['intrinsic_board'],'extrinsics':request['extrinsic_board']},
        'camera_regions':request['camera_regions'],'capture_manifest':'capture/manifest.json',
        'coordinate_contract':'coordinate_contract.json','render_truth_not_solver_input':'capture/render_truth.json',
        'pipeline_groups':specs,'skipped_graphdeco_stages':['prepare','capture','compose']})
    for i,spec in enumerate(specs):
        progress(f'运行原版 1.0.0 内参/条件外参：组 {i+1}/{len(specs)}')
        try:runs.append(Pipeline(spec,base_dir=directory).run(execute=True))
        except (ValueError,RuntimeError,cv2.error,OSError) as exc:failures.append(f'组 {i}: {exc}')
        finally:
            for kind in ('intrinsics','extrinsics'):
                source=root/'groups'/str(i)/kind
                if source.exists():shutil.copytree(source,root/kind,dirs_exist_ok=True)
    bev_runs=[]
    for model in sorted({scene.cameras[n].model for n in request['cameras']}):
        names=[n for n in request['cameras'] if scene.cameras[n].model==model]
        if not all((root/'extrinsics'/model/n/'extrinsics.json').is_file() for n in names):continue
        progress(f'原版 BEV 候选配对/重投影：{model}')
        # Reuse original candidate selection and blending, but NOT its default
        # centered fisheye-circle mask: current lenses may have arbitrary K.
        from calibration_pipeline.bev import prepare_bev_spec,stitch_bev
        b=scene.bev
        try:
            spec=prepare_bev_spec(root/'intrinsics',root/'extrinsics',root/'images/extrinsics',
                model=model,channels=names,grid={'x_min_m':b.x_min,'x_max_m':b.x_max,
                    'y_min_m':b.y_min,'y_max_m':b.y_max,
                    'meters_per_pixel':(b.x_max-b.x_min)/b.height,'plane_z_m':0.})
            # Direct PNG raster, not the old composed inscribed-circle lens.
            # Saved validity is PUBLIC acquisition mask, not K/D or extrinsics.
            # Public max_angle describes the available hemisphere, not a GT fit.
            for camera in spec['cameras']:
                camera['image_mask']='full'
                mask=directory/'capture/valid'/f"{camera['id']}.png"
                camera['source_valid_mask']=str(mask)
                camera['metadata']['source_valid_mask_sha256']=digest(mask)
                camera['metadata']['direct_lens_mask_policy']='Full PNG; invalid capture pixels are explicit zero coverage; see source_valid mask.'
                camera['metadata']['source_circle_policy']='not used for direct-lens acquisition'
            put(directory/'solver_inputs'/f'bev_{model}.json',spec)
            bev_runs.append(stitch_bev(spec,root/'bev'/model,execute=True,base_dir=directory))
        except (ValueError,RuntimeError,cv2.error,OSError) as exc:failures.append(f'BEV {model}: {exc}')
    progress('仅在求解结束后比较 Ground Truth')
    comparison=evaluate(scene,request,directory)
    payload={'adapter':'direct_lens_png_v1','runs':runs,'bev_runs':bev_runs,'failures':failures,
        'ground_truth_comparison':comparison,'calibration_accepted':False}
    put(directory/'scene_pipeline_result.json',payload)
    if failures:raise RuntimeError('标定存在未完成阶段：'+'；'.join(failures))
    return payload
