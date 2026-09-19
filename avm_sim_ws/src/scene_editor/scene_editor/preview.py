"""Asynchronous sensor preview, independent of UI redraws and orbit camera."""
from __future__ import annotations
import time
import numpy as np
from gs_renderer.worker import SensorWorker
from avm_stitcher.projection import AvmStitcher


class PreviewDisplay:
    def __init__(self, server):
        self.server,self.images = server,{}
        with server.gui.add_folder("Camera Preview", expand_by_default=False):
            self.status = server.gui.add_markdown("Waiting for sensor images")
            self.container = server.gui.add_folder("Virtual cameras")
        with server.gui.add_folder("AVM Preview", expand_by_default=False):
            server.gui.add_markdown("From four sensor images | +X up, +Y left")
            self.avm = server.gui.add_image(np.zeros((128,128,3),np.uint8),label="AVM")

    def update(self, images: dict[str,np.ndarray], avm: np.ndarray | None = None) -> None:
        for name in list(self.images):
            if name not in images:
                self.images.pop(name).remove()
        for name,rgb in images.items():
            if name not in self.images:
                with self.container:
                    self.images[name] = self.server.gui.add_image(rgb,label=name,format="jpeg")
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
        with server.gui.add_folder("Rendering", expand_by_default=False):
            self.enabled = server.gui.add_checkbox("Live sensor rendering", initial_value=True)

    def tick(self) -> None:
        revision,scene = self.store.snapshot()
        response = self.worker.poll()
        if response:
            done,rendered_scene,results,error = response
            self.latest_revision = done
            if error:
                self.status.content = f"**Sensor error:**\n\n{error.splitlines()[-1]}"
            else:
                images = {n:r.rgb for n,r in results.items()}
                try:
                    avm = self.stitcher.stitch(rendered_scene,images).rgb
                    warning = ""
                except ValueError as exc:
                    avm,warning = np.zeros((128,128,3),np.uint8),str(exc)
                self.update(images,avm)
                self.status.content = f"{self.backend} | revision {done} | {time.monotonic()-self.last_submit:.2f}s / batch" + (f"\n{warning}" if warning else "")
        if self.enabled.value and not self.worker.busy and revision != self.latest_revision and time.monotonic()-self.last_submit >= self.interval:
            self.last_submit = time.monotonic()
            self.worker.submit(revision,scene)

    def close(self) -> None:
        self.worker.close()
