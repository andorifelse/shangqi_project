"""Asynchronous sensor preview, independent of UI redraws and orbit camera."""
from __future__ import annotations
import time
from collections import OrderedDict
import numpy as np
from gs_renderer.worker import SensorWorker
from .ui_text import camera_name, error_text, BACKEND_LABELS
from avm_stitcher.projection import AvmStitcher


class SensorPreviewBuffer:
    """Only expose complete camera batches rendered for the current revision."""
    def __init__(self):
        self.batches = OrderedDict()

    def batch(self, stamp):
        batch = self.batches.setdefault(stamp, {"images": {}})
        while len(self.batches) > 12:
            self.batches.popitem(last=False)
        return batch

    def metadata(self, stamp, revision, names):
        self.batch(stamp).update(revision=revision, names=tuple(names))

    def image(self, stamp, name, rgb):
        self.batch(stamp)["images"][name] = rgb

    def ready(self, revision):
        complete = []
        for stamp, batch in list(self.batches.items()):
            rendered = batch.get("revision")
            if rendered is not None and rendered < revision:
                del self.batches[stamp]
            elif rendered == revision and all(n in batch["images"] for n in batch["names"]):
                complete.append(stamp)
        if not complete:
            return None
        stamp = max(complete)
        batch = self.batches.pop(stamp)
        for old in list(self.batches):
            if old <= stamp:
                del self.batches[old]
        return stamp, {n: batch["images"][n] for n in batch["names"]}


class PreviewDisplay:
    def __init__(self, server):
        self.server,self.images = server,{}
        with server.gui.add_folder("相机预览", expand_by_default=False):
            self.status = server.gui.add_markdown("等待相机图像")
            self.container = server.gui.add_folder("虚拟相机")
        with server.gui.add_folder("环视预览", expand_by_default=False):
            server.gui.add_markdown("四路图像融合；图中上方为车辆前方，左侧为车辆左方。")
            self.avm = server.gui.add_image(np.zeros((128,128,3),np.uint8),label="环视鸟瞰图")

    def update(self, images: dict[str,np.ndarray], avm: np.ndarray | None = None) -> None:
        for name in list(self.images):
            if name not in images:
                self.images.pop(name).remove()
        order = {n: i for i, n in enumerate(("front", "rear", "left", "right"))}
        for name in sorted(images, key=lambda n: (order.get(n, 4), n)):
            rgb = images[name]
            if name not in self.images:
                with self.container:
                    self.images[name] = self.server.gui.add_image(rgb,label=camera_name(name),format="jpeg")
            else:
                self.images[name].image = rgb
        if avm is not None:
            self.avm.image = avm


class Preview(PreviewDisplay):
    def __init__(self, server, store, backend: str = "cpu", hz: float = 2.):
        super().__init__(server)
        self.store,self.backend = store,backend
        self.worker,self.stitcher = SensorWorker(backend),AvmStitcher()
        self.latest_revision,self.last_submit,self.interval = -1,0.,1/max(hz,.1)
        with server.gui.add_folder("渲染设置", expand_by_default=False):
            self.enabled = server.gui.add_checkbox("实时渲染", initial_value=True)

    def tick(self) -> None:
        revision,scene = self.store.snapshot()
        response = self.worker.poll()
        if response:
            done,rendered_scene,results,error = response
            self.latest_revision = done
            if error:
                self.status.content = f"**相机渲染失败：**\n\n{error_text(error.splitlines()[-1])}"
            else:
                images = {n:r.rgb for n,r in results.items()}
                try:
                    avm = self.stitcher.stitch(rendered_scene,images).rgb
                    warning = ""
                except ValueError as exc:
                    avm,warning = np.zeros((128,128,3),np.uint8),error_text(exc)
                self.update(images,avm)
                self.status.content = f"{BACKEND_LABELS.get(self.backend,self.backend)} · 场景版本 {done} · 每批 {time.monotonic()-self.last_submit:.2f} 秒" + (f"\n{warning}" if warning else "")
        if self.enabled.value and not self.worker.busy and revision != self.latest_revision and time.monotonic()-self.last_submit >= self.interval:
            self.last_submit = time.monotonic()
            self.worker.submit(revision,scene)

    def close(self) -> None:
        self.worker.close()
