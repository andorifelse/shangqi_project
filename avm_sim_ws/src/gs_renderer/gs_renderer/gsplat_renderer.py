"""Native gsplat 1.5.3 adapter: grouped batched pinhole/fisheye rendering.

CUDA path is opt-in until tools/render_smoke.py --backend gsplat succeeds.
Ftheta uses the ray reference backend because its calibration polynomial is
not identical to gsplat's pixel-space FThetaCameraDistortionParameters.
"""
from __future__ import annotations
from collections import defaultdict
import inspect
import warnings
import numpy as np
from scene_manager.models import Scene
from scene_manager.camera_models import camera_rays
from scene_manager.transforms import inverse
from .gaussian import GaussianScene
from .types import RenderResult
from .cpu_renderer import CpuRayRenderer


class GsplatRenderer:
    name = "gsplat CUDA"

    def __init__(self, gaussian: GaussianScene):
        try:
            import torch
            from gsplat import rasterization
        except ImportError as exc:
            raise RuntimeError("gsplat backend requires PyTorch CUDA and gsplat; see INSTALL.md. Use --backend cpu for reference.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA; use --backend cpu or repair the CUDA environment")
        required = {"camera_model","with_ut","with_eval3d","radial_coeffs","tangential_coeffs"}
        if not required.issubset(inspect.signature(rasterization).parameters):
            raise RuntimeError("Installed gsplat lacks native distorted camera API; install gsplat==1.5.3")
        self.torch, self.rasterization = torch, rasterization
        self.tensor = lambda a: torch.as_tensor(a,dtype=torch.float32,device="cuda")
        # Upload Gaussian arrays only when scene geometry changes.
        self.means,self.quats,self.scales,self.opacities,self.colors = [
            self.tensor(a) for a in (gaussian.means,gaussian.quats,gaussian.scales,gaussian.opacities,gaussian.colors)]
        self.fallback = CpuRayRenderer(gaussian)
        self.warned_ftheta = False

    def render_batch(self, scene: Scene) -> dict[str,RenderResult]:
        groups = defaultdict(list)
        outputs = {}
        for name,camera in scene.cameras.items():
            if camera.model == "ftheta":
                if not self.warned_ftheta:
                    warnings.warn("Ftheta uses CPU ray fallback; native gsplat FTheta calibration adapter is pending.")
                    self.warned_ftheta = True
                outputs[name] = self.fallback.render(camera,scene.camera_optical_pose(name))
            else:
                groups[(camera.model,camera.width,camera.height)].append(name)
        for (model,width,height),names in groups.items():
            cameras = [scene.cameras[n] for n in names]
            d = np.array([c.D for c in cameras])
            kwargs = {"radial_coeffs":self.tensor(d)} if model=="fisheye" else {
                "radial_coeffs":self.tensor(np.column_stack([d[:,0],d[:,1],d[:,4],np.zeros((len(d),3))])),
                "tangential_coeffs":self.tensor(d[:,2:4])}
            # gsplat samples at u+.5,v+.5. Our/OpenCV integer pixel centers require
            # principal point +.5 when handing K to the CUDA rasterizer.
            ks = np.array([c.matrix for c in cameras])
            ks[:,:2,2] += .5
            with self.torch.inference_mode():
                rgbd,alpha,_ = self.rasterization(
                    means=self.means,quats=self.quats,scales=self.scales,
                    opacities=self.opacities,colors=self.colors,
                    viewmats=self.tensor(np.array([inverse(scene.camera_optical_pose(n)) for n in names])),
                    Ks=self.tensor(ks),width=width,height=height,
                    camera_model=model,with_ut=True,with_eval3d=True,packed=False,
                    render_mode="RGB+ED",backgrounds=self.tensor(np.tile([.055,.07,.095],(len(names),1))),
                    **kwargs)
                # One batch transfer at the current CPU overlay/ROS boundary.
                rgbd,alpha = rgbd.cpu().numpy(),alpha.cpu().numpy()[...,0]
            for i,(name,camera) in enumerate(zip(names,cameras)):
                rays,valid = camera_rays(camera)
                # v1.5.3 RGB+ED is expected optical z, NOT ray distance.
                depth = rgbd[i,...,3]/np.maximum(rays[...,2],1e-6)
                depth[(alpha[i]<.01)|~valid] = np.inf
                rgb = np.clip(rgbd[i,...,:3]*255,0,255).astype(np.uint8)
                rgb[~valid] = 0
                outputs[name] = RenderResult(rgb,depth.astype(np.float32),alpha[i],valid)
        return outputs
