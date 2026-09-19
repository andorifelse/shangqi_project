"""Interactive editing, with all callbacks serialized onto the editor tick."""
from __future__ import annotations
from queue import SimpleQueue
import numpy as np
from scene_manager.transforms import (pose_matrix, matrix_pose, inverse,
    wxyz_from_matrix, matrix_from_wxyz_position)
from .objects import ObjectView


class Editor:
    def __init__(self, server, store):
        self.server, self.store = server, store
        self.events = SimpleQueue()
        self.selected = "base_link"
        self.revision = -1
        self.gizmo = None
        self.gizmo_offset = np.zeros(3)
        self.objects = ObjectView(server, lambda name: self.events.put(("select", name)))
        server.gui.add_markdown("# AVM Simulation\nMeters  |  Z up  |  RPY in degrees\n\nOrbit: left drag  |  Pan: right drag  |  Zoom: wheel")
        self.status = server.gui.add_markdown("Ready")
        with server.gui.add_folder("Scene Objects"):
            self.tree = server.gui.add_dropdown("Selected", options=("base_link",), initial_value="base_link")
            self.tree.on_update(lambda e: self.events.put(("select", self.tree.value)) if e.client else None)
            for label, kind in [("Add Vehicle", "vehicle"), ("Add Camera", "camera"), ("Add Board", "board")]:
                btn = server.gui.add_button(label)
                btn.on_click(lambda _, k=kind: self.events.put(("add", k)))
            server.gui.add_button("Delete selected").on_click(lambda _: self.events.put(("delete", None)))
        with server.gui.add_folder("Transform"):
            self.parent_label = server.gui.add_markdown("Parent: world")
            self.position = [server.gui.add_number(a, 0., step=.05) for a in ("X", "Y", "Z")]
            self.rotation = [server.gui.add_number(a, 0., step=1.) for a in ("Roll", "Pitch", "Yaw")]
            for handle in self.position + self.rotation:
                handle.on_update(self._numeric)
            self.mode = server.gui.add_dropdown("Gizmo", options=("Translate + Rotate", "Translate", "Rotate"))
            self.mode.on_update(lambda e: self.events.put(("select", self.selected)) if e.client else None)
        self._build_properties()
        self.tick()

    def _numeric(self, event) -> None:
        if event.client:
            values = [h.value for h in self.position] + np.deg2rad([h.value for h in self.rotation]).tolist()
            self.events.put(("command", {"action":"pose", "id":self.selected, "pose":values}))

    def _build_properties(self) -> None:
        with self.server.gui.add_folder("Camera parameters", expand_by_default=False):
            self.cam_model = self.server.gui.add_dropdown("Model", options=("fisheye","pinhole","ftheta"))
            self.cam_width = self.server.gui.add_number("Width", 320, min=16, max=4096, step=1)
            self.cam_height = self.server.gui.add_number("Height", 240, min=16, max=4096, step=1)
            self.intrinsic = [self.server.gui.add_number(k, v, step=1.) for k,v in
                              zip(("fx","fy","cx","cy"), (110.,110.,159.5,119.5))]
            self.distortion = self.server.gui.add_text("D (comma separated)", "0,0,0,0")
            self.max_angle = self.server.gui.add_number("Max ray angle (deg)", 88.8, step=.1)
            self.server.gui.add_button("Apply camera parameters").on_click(lambda _: self.events.put(("camera", None)))
        with self.server.gui.add_folder("Board parameters", expand_by_default=False):
            self.rows = self.server.gui.add_number("Rows (squares)", 6, min=1, max=100, step=1)
            self.columns = self.server.gui.add_number("Columns (squares)", 8, min=1, max=100, step=1)
            self.square = self.server.gui.add_number("Square size (m)", .15, min=.001, step=.01)
            self.board_size = self.server.gui.add_markdown("Physical size = columns/rows x square size")
            self.server.gui.add_button("Apply board parameters").on_click(lambda _: self.events.put(("board", None)))

    def _select(self, scene) -> None:
        if self.gizmo:
            self.gizmo.remove()
            self.gizmo = None
        if self.selected not in scene.object_ids():
            self.selected = next(iter(scene.object_ids()), "")
        self._shown_selection = self.selected
        if not self.selected:
            return
        obj = scene.object(self.selected)
        self.tree.value = self.selected
        self.parent_label.content = f"Parent: {getattr(obj, 'parent', 'world')}  |  camera values are relative to parent"
        for h,v in zip(self.position + self.rotation, obj.pose[:3]+np.rad2deg(obj.pose[3:]).tolist()):
            h.value = float(v)
        if self.selected in scene.cameras:
            c = scene.cameras[self.selected]
            self.cam_model.value, self.cam_width.value, self.cam_height.value = c.model, c.width, c.height
            for h,v in zip(self.intrinsic, [c.K[0],c.K[4],c.K[2],c.K[5]]):
                h.value = float(v)
            self.distortion.value = ",".join(str(d) for d in c.D)
            self.max_angle.value = float(np.rad2deg(c.max_angle))
        if self.selected in scene.boards:
            b = scene.boards[self.selected]
            self.rows.value, self.columns.value, self.square.value = b.rows,b.columns,b.square_size
            self.board_size.content = f"{b.width:.3f} m x {b.height:.3f} m"
        t = scene.world_pose(self.selected)
        self.gizmo_offset = np.array([0., 0., 2.0 if self.selected == "base_link" else .4])
        self.gizmo = self.server.scene.add_transform_controls("/gizmo", scale=115., fixed=True, line_width=5.,
            position=t[:3,3]+self.gizmo_offset, wxyz=wxyz_from_matrix(t),
            disable_axes=self.mode.value == "Rotate", disable_sliders=self.mode.value == "Rotate",
            disable_rotations=self.mode.value == "Translate", depth_test=True)
        self.gizmo.on_update(self._drag)

    async def _drag(self, event) -> None:
        if event.client and self.gizmo:
            self.events.put(("drag", (self.selected, np.array(event.target.wxyz), np.array(event.target.position)-self.gizmo_offset)))

    def _handle(self, kind, value) -> None:
        _, scene = self.store.snapshot()
        if kind == "select":
            self.selected = value
            self._select(scene)
        elif kind == "command":
            self.store.command(value)
        elif kind == "drag":
            name, q, p = value
            t = matrix_from_wxyz_position(q, p)
            obj = scene.object(name)
            if getattr(obj, "parent", None) == "base_link":
                t = inverse(scene.world_pose("base_link")) @ t
            self.store.command({"action":"pose","id":name,"pose":matrix_pose(t)})
        elif kind == "add":
            if value == "vehicle":
                self.store.command({"action":"add_vehicle"})
                self.selected = "base_link"
            else:
                if value == "camera":
                    names = ["front","rear","left","right","extra_left","extra_right"]
                    name = next((n for n in names if n not in scene.object_ids()), None)
                    if name is None:
                        raise ValueError("Six camera limit reached")
                    props = {"parent":"base_link" if scene.vehicle else "world"}
                else:
                    idx = 1
                    while f"board_{idx:03d}" in scene.object_ids():
                        idx += 1
                    name, props = f"board_{idx:03d}", {}
                self.store.command({"action":f"add_{value}","name":name,"properties":props})
                self.selected = name
        elif kind == "delete":
            self.store.command({"action":"delete","id":self.selected})
            self.selected = ""
        elif kind == "camera":
            if self.selected not in scene.cameras:
                raise ValueError("Select a camera first")
            fx,fy,cx,cy = [h.value for h in self.intrinsic]
            self.store.command({"action":"properties","id":self.selected,"properties":{
                "model":self.cam_model.value, "width":int(self.cam_width.value), "height":int(self.cam_height.value),
                "K":[fx,0,cx,0,fy,cy,0,0,1], "D":[float(v.strip()) for v in self.distortion.value.split(",")],
                "max_angle":float(np.deg2rad(self.max_angle.value))}})
        elif kind == "board":
            if self.selected not in scene.boards:
                raise ValueError("Select a board first")
            self.store.command({"action":"properties","id":self.selected,"properties":{
                "rows":int(self.rows.value),"columns":int(self.columns.value),"square_size":self.square.value}})

    def tick(self) -> None:
        while not self.events.empty():
            try:
                self._handle(*self.events.get())
                self.status.content = "Ready"
            except (ValueError, KeyError, OSError) as exc:
                self.status.content = f"**Edit rejected:** {exc}"
        revision, scene = self.store.snapshot()
        if revision == self.revision:
            return
        try:
            self.objects.sync(scene)
        except (ValueError, OSError) as exc:
            self.status.content = f"**Object display error:** {exc}"
            self.revision = revision
            return
        self.tree.options = tuple(scene.object_ids()) or ("(empty)",)
        # Reuse gizmo while dragging; recreating it breaks continuous mouse capture.
        ids_changed = self.selected not in scene.object_ids()
        if self.gizmo is None or ids_changed or getattr(self, "_shown_selection", None) != self.selected:
            self._select(scene)
        elif self.selected:
            t = scene.world_pose(self.selected)
            self.gizmo.position, self.gizmo.wxyz = t[:3,3]+self.gizmo_offset, wxyz_from_matrix(t)
            obj = scene.object(self.selected)
            for h,v in zip(self.position+self.rotation, obj.pose[:3]+np.rad2deg(obj.pose[3:]).tolist()):
                h.value = float(v)
        self._shown_selection = self.selected
        self.revision = revision
