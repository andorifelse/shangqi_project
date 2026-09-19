"""Scene controls shared by standalone and ROS editor."""
from queue import SimpleQueue
from pathlib import Path
import numpy as np
from gs_renderer.gaussian import GaussianScene


class SceneFiles:
    def __init__(self, server, store, status, default_path: Path):
        self.server, self.store, self.status = server, store, status
        self.events = SimpleQueue()
        self.gaussian_key = None
        self.gaussian_handle = None
        with server.gui.add_folder("Scene files", expand_by_default=False):
            self.path = server.gui.add_text("Scene YAML", str(default_path))
            server.gui.add_button("Save Scene").on_click(lambda _: self.events.put({"action":"save","path":self.path.value}))
            server.gui.add_button("Load Scene").on_click(lambda _: self.events.put({"action":"load","path":self.path.value}))
            self.ply = server.gui.add_text("Gaussian PLY path", "")
            server.gui.add_button("Load Gaussian PLY").on_click(lambda _: self.events.put({"action":"gaussian","path":self.ply.value}))
            self.glb = server.gui.add_text("Vehicle GLB / GLTF / 3DGS PLY path", "")
            server.gui.add_button("Set vehicle asset").on_click(lambda _: self.events.put({
                "action":"properties","id":"base_link","properties":{"asset":str(Path(self.glb.value).expanduser().resolve())}}))
        self.tick()

    def tick(self) -> None:
        while not self.events.empty():
            command = self.events.get()
            try:
                if command["action"] == "properties" and "asset" in command["properties"]:
                    asset = command["properties"]["asset"]
                    if Path(asset).suffix.lower() == ".ply":
                        GaussianScene.load(asset)
                    else:
                        import trimesh
                        trimesh.load_scene(asset)
                if command["action"] == "gaussian":
                    command["path"] = str(Path(command["path"]).expanduser().resolve())
                    GaussianScene.load(command["path"])
                self.store.command(command)
                self.status.content = f"Scene {command['action']} completed"
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.status.content = f"**Scene action failed:** {exc}"
        _, scene = self.store.snapshot()
        key = (scene.gaussian_file, repr(scene.coordinate))
        if key != self.gaussian_key:
            try:
                if scene.gaussian_file:
                    gs = GaussianScene.load(scene.gaussian_file, scene.coordinate.scale,
                                            np.array(scene.coordinate.T_world_from_gs))
                    if self.gaussian_handle:
                        self.gaussian_handle.remove()
                    self.gaussian_handle = self.server.scene.add_gaussian_splats("/gaussian",
                        centers=gs.means, covariances=gs.covariances,
                        rgbs=gs.colors, opacities=gs.opacities[:,None])
                elif self.gaussian_handle:
                    self.gaussian_handle.remove()
                    self.gaussian_handle = None
                self.gaussian_key = key
            except (OSError, ValueError) as exc:
                self.status.content = f"**Gaussian load failed:** {exc}"
                self.gaussian_key = key  # Do not flood errors; retry after next asset change.
