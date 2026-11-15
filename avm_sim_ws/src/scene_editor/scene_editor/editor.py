"""Interactive editing, with all callbacks serialized onto the editor tick."""
from __future__ import annotations
from queue import SimpleQueue
import numpy as np
from scene_manager.transforms import (pose_matrix, matrix_pose, inverse,
    wxyz_from_matrix, matrix_from_wxyz_position)
from .objects import ObjectView
from .ui_text import error_text, MODEL_LABELS, MODEL_VALUES, object_label, object_id


class Editor:
    def __init__(self, server, store):
        self.server, self.store = server, store
        self.events = SimpleQueue()
        self.selected = "base_link"
        self.revision = -1
        self._view_key = None
        self.gizmo = None
        self.gizmo_offset = np.zeros(3)
        self._gizmo_generation = 0
        self._dragging = False
        self.objects = ObjectView(server, lambda name: self.events.put(("select", name)))
        server.gui.add_markdown("# 环视标定仿真\n位置：米 · 角度：度 · Z 轴向上\n左键旋转 · 右键平移 · 滚轮缩放")
        self.status = server.gui.add_markdown("就绪")
        with server.gui.add_folder("场景对象"):
            self.tree = server.gui.add_dropdown("当前对象", options=(object_label("base_link"),))
            self.tree.on_update(lambda e: self.events.put(("select", object_id(self.tree.value))) if e.client else None)
            for label, kind in [("添加车辆", "vehicle"), ("添加相机", "camera"), ("添加标定板", "board")]:
                btn = server.gui.add_button(label)
                btn.on_click(lambda _, k=kind: self.events.put(("add", k)))
            server.gui.add_button("删除当前对象").on_click(lambda _: self.events.put(("delete", None)))
        with server.gui.add_folder("位姿调整"):
            self.parent_label = server.gui.add_markdown("参考坐标系：世界")
            self.position = [server.gui.add_number(a, 0., step=.05) for a in ("X", "Y", "Z")]
            self.rotation = [server.gui.add_number(a, 0., step=1.) for a in ("横滚角", "俯仰角", "偏航角")]
            for handle in self.position + self.rotation:
                handle.on_update(self._numeric)
            self.mode = server.gui.add_dropdown("拖拽模式", options=("平移与旋转", "平移", "旋转"))
            self.mode.on_update(lambda e: self.events.put(("select", self.selected)) if e.client else None)
        self._build_properties()
        self.tick()

    def _numeric(self, event) -> None:
        if event.client:
            values = [h.value for h in self.position] + np.deg2rad([h.value for h in self.rotation]).tolist()
            self.events.put(("command", {"action":"pose", "id":self.selected, "pose":values}))

    def _build_properties(self) -> None:
        with self.server.gui.add_folder("相机参数", expand_by_default=False):
            self.cam_model = self.server.gui.add_dropdown("镜头模型", options=("鱼眼","针孔","等距多项式"))
            self.cam_width = self.server.gui.add_number("图像宽度（像素）", 320, min=16, max=4096, step=1)
            self.cam_height = self.server.gui.add_number("图像高度（像素）", 240, min=16, max=4096, step=1)
            self.intrinsic = [self.server.gui.add_number(k, v, step=1.) for k,v in
                              zip(("fx","fy","cx","cy"), (110.,110.,159.5,119.5))]
            self.distortion = self.server.gui.add_text("畸变系数（逗号分隔）", "0,0,0,0")
            self.max_angle = self.server.gui.add_number("最大光线半角（度）", 88.8, step=.1)
            self.server.gui.add_button("应用相机参数").on_click(lambda _: self.events.put(("camera", None)))
        with self.server.gui.add_folder("标定板参数", expand_by_default=False):
            self.rows = self.server.gui.add_number("方格行数", 6, min=1, max=100, step=1)
            self.columns = self.server.gui.add_number("方格列数", 8, min=1, max=100, step=1)
            self.square = self.server.gui.add_number("方格边长（米）", .15, min=.001, step=.01)
            self.board_size = self.server.gui.add_markdown("物理尺寸由行列数和格长确定")
            self.server.gui.add_button("应用棋盘参数").on_click(lambda _: self.events.put(("board", None)))
            self.board_asset = self.server.gui.add_text("高斯板 PLY 路径（留空为虚拟板）", "")
            self.board_asset_scale = self.server.gui.add_number("高斯板缩放倍数", 1., min=.000001, step=.01)
            self.board_asset_position = [self.server.gui.add_number(f"模型局部 {a}（米）", 0., step=.01) for a in ("X","Y","Z")]
            self.board_asset_rotation = [self.server.gui.add_number(f"模型局部 {a}（度）", 0., step=1.) for a in ("横滚角","俯仰角","偏航角")]
            self.server.gui.add_markdown("局部对齐校正模型原点和方向；高斯板大小由缩放控制，格长须与实际模型一致。")
            self.server.gui.add_button("应用标定板模型").on_click(lambda _: self.events.put(("board_asset", None)))
            self.server.gui.add_button("切换为虚拟棋盘").on_click(lambda _: self.events.put(("board_virtual", None)))

    def _select(self, scene) -> None:
        if self.gizmo:
            self.gizmo.remove()
            self.gizmo = None
        self._dragging = False
        if self.selected not in scene.object_ids():
            self.selected = next(iter(scene.object_ids()), "")
        self._shown_selection = self.selected
        if not self.selected:
            return
        obj = scene.object(self.selected)
        self.tree.value = object_label(self.selected)
        self.parent_label.content = f"参考坐标系：{'车体' if getattr(obj, 'parent', 'world') == 'base_link' else '世界'}（相机位姿相对于参考系）"
        for h,v in zip(self.position + self.rotation, obj.pose[:3]+np.rad2deg(obj.pose[3:]).tolist()):
            h.value = float(v)
        if self.selected in scene.cameras:
            c = scene.cameras[self.selected]
            self.cam_model.value, self.cam_width.value, self.cam_height.value = MODEL_LABELS[c.model], c.width, c.height
            for h,v in zip(self.intrinsic, [c.K[0],c.K[4],c.K[2],c.K[5]]):
                h.value = float(v)
            self.distortion.value = ",".join(str(d) for d in c.D)
            self.max_angle.value = float(np.rad2deg(c.max_angle))
        if self.selected in scene.boards:
            self._sync_board_fields(scene.boards[self.selected])
        t = scene.world_pose(self.selected)
        self.gizmo_offset = np.array([0., 0., 2.0 if self.selected == "base_link" else .4])
        # A fresh wire name prevents delayed messages for a removed control
        # from being routed to a different camera's newly created control.
        self._gizmo_generation += 1
        self.gizmo = self.server.scene.add_transform_controls(f"/gizmo/{self._gizmo_generation}", scale=115., fixed=True, line_width=5.,
            position=t[:3,3]+self.gizmo_offset, wxyz=wxyz_from_matrix(t),
            disable_axes=self.mode.value == "旋转", disable_sliders=self.mode.value == "旋转",
            disable_rotations=self.mode.value == "平移", depth_test=True)
        self.gizmo.on_update(self._drag)

    def _sync_board_fields(self, board) -> None:
        # Refresh only when authoritative properties change. This handles ROS
        # asynchronous service echoes without overwriting an un-applied edit
        # on every pose tick.
        key = (board.id, board.rows, board.columns, board.square_size,
               board.asset, board.asset_scale, tuple(board.asset_pose))
        self.rows.value, self.columns.value, self.square.value = board.rows,board.columns,board.square_size
        self.board_size.content = f"{board.width:.3f} m x {board.height:.3f} m"
        self.board_asset.value, self.board_asset_scale.value = board.asset,board.asset_scale
        for h,v in zip(self.board_asset_position+self.board_asset_rotation,
                       board.asset_pose[:3]+np.rad2deg(board.asset_pose[3:]).tolist()):
            h.value = float(v)
        self._board_fields_key = key

    async def _drag(self, event) -> None:
        if not event.client or event.target is not self.gizmo:
            return
        phase = getattr(event, "phase", "update")
        self._dragging = phase != "end"
        if phase == "start":
            return
        # Snapshot identity and values before the editor consumes the queue.
        self.events.put(("drag", (self.selected, np.array(event.target.wxyz).copy(),
                                  np.array(event.target.position).copy()-self.gizmo_offset)))

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
                        raise ValueError("最多支持六个相机")
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
                raise ValueError("请先选择相机")
            fx,fy,cx,cy = [h.value for h in self.intrinsic]
            self.store.command({"action":"properties","id":self.selected,"properties":{
                "model":MODEL_VALUES[self.cam_model.value], "width":int(self.cam_width.value), "height":int(self.cam_height.value),
                "K":[fx,0,cx,0,fy,cy,0,0,1], "D":[float(v.strip()) for v in self.distortion.value.split(",")],
                "max_angle":float(np.deg2rad(self.max_angle.value))}})
        elif kind == "board":
            if self.selected not in scene.boards:
                raise ValueError("请先选择标定板")
            self.store.command({"action":"properties","id":self.selected,"properties":{
                "rows":int(self.rows.value),"columns":int(self.columns.value),"square_size":self.square.value}})
        elif kind in ("board_asset", "board_virtual"):
            if self.selected not in scene.boards:
                raise ValueError("请先选择标定板")
            from pathlib import Path
            from gs_renderer.gaussian import GaussianScene
            raw_path = self.board_asset.value.strip() if kind == "board_asset" else ""
            asset = str(Path(raw_path).expanduser().resolve()) if raw_path else ""
            if asset:
                GaussianScene.load(asset)  # Validate before changing authoritative state.
            self.store.command({"action":"properties", "id":self.selected, "properties":{
                "asset":asset, "asset_scale":float(self.board_asset_scale.value),
                "asset_pose":[h.value for h in self.board_asset_position]+
                    np.deg2rad([h.value for h in self.board_asset_rotation]).tolist()}})
            self._select(self.store.snapshot()[1])

    def tick(self) -> None:
        while not self.events.empty():
            try:
                self._handle(*self.events.get())
                self.status.content = "就绪"
            except (ValueError, KeyError, OSError) as exc:
                self.status.content = f"**修改未生效：** {error_text(exc)}"
        revision, scene = self.store.snapshot()
        view_key = (revision, tuple((name, tuple(scene.object(name).pose)) for name in scene.object_ids()))
        if view_key == self._view_key:
            return
        try:
            self.objects.sync(scene)
        except (ValueError, OSError) as exc:
            self.status.content = f"**对象显示失败：** {error_text(exc)}"
            self.revision = revision
            self._view_key = view_key
            return
        self.tree.options = tuple(object_label(n) for n in scene.object_ids()) or ("（空场景）",)
        # Reuse gizmo while dragging; recreating it breaks continuous mouse capture.
        ids_changed = self.selected not in scene.object_ids()
        if self.gizmo is None or ids_changed or getattr(self, "_shown_selection", None) != self.selected:
            self._select(scene)
        elif self.selected:
            t = scene.world_pose(self.selected)
            # Authoritative ROS state can trail the pointer by several frames.
            # Repositioning PivotControls mid-drag changes its drag reference.
            if not self._dragging:
                self.gizmo.position, self.gizmo.wxyz = t[:3,3]+self.gizmo_offset, wxyz_from_matrix(t)
            obj = scene.object(self.selected)
            for h,v in zip(self.position+self.rotation, obj.pose[:3]+np.rad2deg(obj.pose[3:]).tolist()):
                h.value = float(v)
        self._shown_selection = self.selected
        self.revision = revision
        self._view_key = view_key
        if self.selected in scene.boards:
            b = scene.boards[self.selected]
            key = (b.id,b.rows,b.columns,b.square_size,b.asset,b.asset_scale,tuple(b.asset_pose))
            if key != getattr(self, "_board_fields_key", None):
                self._sync_board_fields(b)
