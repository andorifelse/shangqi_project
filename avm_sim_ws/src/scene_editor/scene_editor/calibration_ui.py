"""Calibration controls shared by standalone and ROS editors."""
from pathlib import Path
from queue import SimpleQueue
import time
import cv2
import numpy as np
from .calibration_jobs import CalibrationJobs,BASELINE,USER_PIPELINE,MULTITILT
from .geometry_ui import BoardGeometryPanel,region_names
from .ui_text import display_text, camera_name, error_text, object_label, object_id, BACKEND_LABELS, BACKEND_VALUES

ALGORITHMS={"OpenCV 棋盘格基线（当前场景）":BASELINE,
            "标定流程 1.0.0":USER_PIPELINE,
            "多倾角采集：鱼眼内参 / 编码板外参":MULTITILT}
ALL_CAMERAS="全部相机"
INPUT_MODES={'当前 GUI 场景（一键采集）':'scene','高级：手工 JSON':'manual','演示：CPU 示例图片':'demo'}


class CalibrationPanel:
    def __init__(self, server, store, backend="cpu", jobs=None):
        self.server,self.store=server,store
        self.jobs=jobs or CalibrationJobs()
        self.events=SimpleQueue();self._ids=None;self._last_poll=0.;self._displayed=None
        self._region_key=None;self._settings_key=None;self.region_controls={}
        self.geometry=BoardGeometryPanel(server,store)
        with server.gui.add_folder("相机标定",expand_by_default=False):
            self.algorithm=server.gui.add_dropdown("标定算法",options=tuple(ALGORITHMS))
            self.info=server.gui.add_markdown("OpenCV：冻结当前场景和相机参数，生成图片再标定，不修改现有相机。")
            self.algorithm.on_update(lambda _:self.events.put("algorithm"))
            self.input_mode=server.gui.add_dropdown('1.0.0 输入来源',options=tuple(INPUT_MODES))
            self.input_mode.on_update(lambda _:self.events.put('algorithm'))
            self.experiment_id=server.gui.add_text("实验名称（留空自动生成）","")
            self.camera=server.gui.add_dropdown("待标定相机",options=(ALL_CAMERAS,))
            self.board=server.gui.add_dropdown("标定板",options=("(无标定板)",))
            self.views=server.gui.add_number("每相机采样数",40,min=8,max=500,step=1)
            self.seed=server.gui.add_number("随机种子",20261020,min=0,max=2**32-1,step=1)
            self.supersample=server.gui.add_number("超采样倍数",2,min=1,max=4,step=1)
            self.backend=server.gui.add_dropdown("渲染方式",options=tuple(BACKEND_VALUES),initial_value=BACKEND_LABELS[backend])
            self.geometry_ok=server.gui.add_checkbox("已核实高斯板为完整矩形棋盘且格长、方向正确",initial_value=False)
            self.intrinsic_board=server.gui.add_dropdown('内参运动小板（完整矩形）',options=('(无)',))
            self.extrinsic_board=server.gui.add_dropdown('外参固定大板（分区／条带）',options=('(无)',))
            self.extrinsic_board.on_update(lambda _:self.events.put('regions'))
            self.holdout=server.gui.add_number('每相机留出图片数',2,min=1,max=30,step=1)
            self.spread_x=server.gui.add_number('小板局部 X 平移半范围（米）',.5,min=0,max=20,step=.05)
            self.spread_y=server.gui.add_number('小板局部 Y 平移半范围（米）',.5,min=0,max=20,step=.05)
            self.yaw_range=server.gui.add_number('小板面内旋转半范围（度）',30.,min=0,max=180,step=1.)
            self.sampling=server.gui.add_dropdown('平面采样方式',options=('自动：逐相机可见平面','当前小板附近局部 XY 范围'))
            self.mapping_folder=server.gui.add_folder('相机与大板区域对应',expand_by_default=True)
            self.save_settings=server.gui.add_button('保存本场景标定设置')
            self.save_settings.on_click(lambda _:self.events.put('save_settings'))
            self.pipeline_config=server.gui.add_text("1.0.0 配置 JSON（仅高级模式）",'')
            self.stage=server.gui.add_text("执行阶段（逗号分隔）","全部")
            self.tilt=server.gui.add_number('小板最大倾角（度）',40.,min=20.,max=55.,step=1.)
            self.distance_min=server.gui.add_number('最小采集距离（米）',1.,min=.4,max=10.,step=.1)
            self.distance_max=server.gui.add_number('最大采集距离（米）',2.4,min=.4,max=10.,step=.1)
            self.run_extrinsics=server.gui.add_checkbox('内参通过后继续编码板外参',initial_value=True)
            self.marker_geometry_file=server.gui.add_text('编码板几何 JSON（留空从资产包查找）','')
            self.run_extrinsics.on_update(lambda _:self.events.put('algorithm'))
            self.start_button=server.gui.add_button("开始标定")
            self.start_button.on_click(lambda _:self.events.put("start"))
            self.cancel_button=server.gui.add_button("取消标定",disabled=True)
            self.cancel_button.on_click(lambda _:self.events.put("cancel"))
            self.download_button=server.gui.add_button("导出结果 JSON",disabled=True)
            self.download_button.on_click(self._download)
            self.status=server.gui.add_markdown("尚未运行")
            self.results=server.gui.add_markdown("尚无标定结果")
            self.preview=server.gui.add_image(np.zeros((64,128,3),np.uint8),label="标定参数鸟瞰图")
            self.preview.visible=False
            self.truth_preview=server.gui.add_image(np.zeros((64,128,3),np.uint8),label="真值参数鸟瞰图（同一组采集图片）")
            self.truth_preview.visible=False
        self._update_algorithm();self.tick()

    def _update_algorithm(self):
        baseline=ALGORITHMS[self.algorithm.value]==BASELINE
        tilted=ALGORITHMS[self.algorithm.value]==MULTITILT
        current=ALGORITHMS[self.algorithm.value]==USER_PIPELINE and INPUT_MODES[self.input_mode.value]=='scene'
        self.views.min=12 if tilted else (8 if baseline else 4)
        self.holdout.min=3 if tilted else 1
        if tilted:
            self.holdout.value=max(3,self.holdout.value)
            self.views.value=max(12,self.views.value)
        manual=not baseline and not tilted and INPUT_MODES[self.input_mode.value]=='manual'
        self.input_mode.visible=not baseline and not tilted
        for h in (self.camera,self.views,self.seed,self.supersample,self.backend):h.visible=baseline or current or tilted
        for h in (self.board,self.geometry_ok):h.visible=baseline
        for h in (self.intrinsic_board,self.extrinsic_board,self.holdout,self.spread_x,self.spread_y,
                  self.yaw_range,self.sampling,self.mapping_folder,self.save_settings):h.visible=current
        for h in (self.pipeline_config,self.stage):h.visible=manual
        for h in (self.tilt,self.distance_min,self.distance_max):h.visible=tilted
        self.run_extrinsics.visible=tilted
        self.marker_geometry_file.visible=tilted and self.run_extrinsics.value
        self.intrinsic_board.visible=current or tilted
        self.holdout.visible=current or tilted
        self.save_settings.visible=current or tilted
        if tilted:
            self.info.content='固定相机，多倾角采集小板；留出图不参与内参拟合。求外参需已知编号和几何的编码板。结果不会自动写回相机。'
            return
        self.info.content=('使用当前场景快照，要求完整矩形棋盘；环车条带不适用。结果不自动覆盖相机参数。' if baseline else
            '固定相机和每路板平面，仅面内移动小板；高斯板须有真实角点定义。固定平面可能导致参数退化。' if current else
            '按配置文件的数据执行；演示模式不使用当前场景。流程完成不代表精度验收通过。')

    def _stage_value(self):
        stages={'全部':'all','场景准备':'prepare','图像采集':'capture','镜头合成':'compose',
                '内参':'intrinsics','外参':'extrinsics','鸟瞰':'bev'}
        value=self.stage.value.strip() or '全部'
        return ','.join(stages.get(v.strip(),v.strip()) for v in value.replace('，',',').split(','))

    def settings(self):
        if ALGORITHMS[self.algorithm.value]==MULTITILT:
            return {'algorithm':MULTITILT,'experiment_id':self.experiment_id.value,
                'cameras':None if self.camera.value==ALL_CAMERAS else [object_id(self.camera.value)],
                'intrinsic_board':self.intrinsic_board.value,'views':int(self.views.value),
                'holdout':int(self.holdout.value),'seed':int(self.seed.value),
                'supersample':int(self.supersample.value),'backend':BACKEND_VALUES[self.backend.value],
                'tilt_deg':float(self.tilt.value),'distance_min':float(self.distance_min.value),
                'distance_max':float(self.distance_max.value),
                'run_extrinsics':self.run_extrinsics.value,
                'marker_geometry_file':self.marker_geometry_file.value.strip()}
        return {"algorithm":ALGORITHMS[self.algorithm.value],"experiment_id":self.experiment_id.value,
            "cameras":None if self.camera.value==ALL_CAMERAS else [object_id(self.camera.value)],
            "board_id":self.board.value,"views":int(self.views.value),"seed":int(self.seed.value),
            "supersample":int(self.supersample.value),"backend":BACKEND_VALUES[self.backend.value],
            "confirm_board_geometry":self.geometry_ok.value,"pipeline_config":self.pipeline_config.value,
            "stage":self._stage_value(),'input_mode':INPUT_MODES[self.input_mode.value],
            'intrinsic_board':self.intrinsic_board.value,'extrinsic_board':self.extrinsic_board.value,
            'holdout':int(self.holdout.value),'spread_x':float(self.spread_x.value),
            'spread_y':float(self.spread_y.value),'yaw_range_deg':float(self.yaw_range.value),
            'sampling':'visible_plane' if self.sampling.value.startswith('自动') else 'around_pose',
            'camera_regions':{n:h.value for n,h in self.region_controls.items() if h.value!='(未配置)'}}

    def _regions(self,scene):
        names=region_names(scene.boards[self.extrinsic_board.value]) if self.extrinsic_board.value in scene.boards else []
        options=('(未配置)',*names)
        key=(tuple(scene.cameras),self.extrinsic_board.value,tuple(names))
        if key==self._region_key:return
        old={n:h.value for n,h in self.region_controls.items()}
        for h in self.region_controls.values():h.remove()
        self.region_controls={}
        with self.mapping_folder:
            for n in scene.cameras:
                v=scene.calibration.get('camera_regions',{}).get(n,old.get(n,'(未配置)'))
                self.region_controls[n]=self.server.gui.add_dropdown(f'{camera_name(n)}对应区域',options=options,
                    initial_value=v if v in options else '(未配置)')
        self._region_key=key

    def _load_saved(self,scene):
        saved=scene.calibration
        self.run_extrinsics.value=saved.get('run_extrinsics', True)
        self.marker_geometry_file.value=saved.get('marker_geometry_file','')
        if saved.get('algorithm') in ALGORITHMS.values():
            self.algorithm.value=next(n for n,v in ALGORITHMS.items() if v==saved['algorithm'])
        for key,control in [('tilt_deg',self.tilt),('distance_min',self.distance_min),('distance_max',self.distance_max)]:
            if key in saved:control.value=saved[key]
        for key,h in (('intrinsic_board',self.intrinsic_board),('extrinsic_board',self.extrinsic_board)):
            if saved.get(key) in scene.boards:h.value=saved[key]
        for key,h in (('views',self.views),('holdout',self.holdout),('seed',self.seed),
                      ('spread_x',self.spread_x),('spread_y',self.spread_y),('yaw_range_deg',self.yaw_range)):
            if key in saved:h.value=saved[key]
        if saved.get('backend') in ('cpu','gsplat'):self.backend.value=BACKEND_LABELS[saved['backend']]
        if 'supersample' in saved:self.supersample.value=saved['supersample']
        if saved.get('sampling')=='around_pose':self.sampling.value='当前小板附近局部 XY 范围'
        elif saved.get('sampling')=='visible_plane':self.sampling.value='自动：逐相机可见平面'
        self._update_algorithm()

    def _download(self,event):
        if not event.client or not self.jobs.directory:return
        path=self.jobs.directory/"result.json"
        if path.is_file():
            event.client.send_file_download(f"{self.jobs.directory.name}_result.json",path.read_bytes())

    def tick(self):
        self.geometry.tick()
        revision,scene=self.store.snapshot()
        ids=(tuple(scene.cameras),tuple(scene.boards))
        if ids!=self._ids:
            self.camera.options=(ALL_CAMERAS,*(object_label(n) for n in ids[0]))
            self.board.options=ids[1] or ("(无标定板)",)
            self.intrinsic_board.options=ids[1] or ('(无)',)
            self.extrinsic_board.options=ids[1] or ('(无)',)
            if len(ids[1])>1:self.extrinsic_board.value=ids[1][1]
            self._load_saved(scene)
            self._ids=ids
        settings_key=repr(scene.calibration)
        if settings_key!=self._settings_key:
            self._load_saved(scene);self._region_key=None;self._settings_key=settings_key
        self._regions(scene)
        while not self.events.empty():
            action=self.events.get()
            try:
                if action=="algorithm":self._update_algorithm()
                elif action=='regions':self._regions(scene)
                elif action=='save_settings':
                    settings=self.settings()
                    settings={k:v for k,v in settings.items() if k not in ('experiment_id','pipeline_config','confirm_board_geometry','stage')}
                    self.store.command({'action':'calibration_settings','settings':settings})
                    self.status.content='标定设置已应用；请保存场景以保留配置。'
                elif action=="start":
                    settings=self.settings()
                    directory=self.jobs.start(self.store,settings)
                    self.status.content=f"后台标定已启动：`{directory}`"
                    self.results.content="正在运行，不会覆盖当前相机参数。"
                    self.preview.visible=False;self.truth_preview.visible=False;self._displayed=None
                elif action=="cancel":self.jobs.cancel()
            except (ValueError,OSError,KeyError,RuntimeError,TypeError) as exc:
                self.status.content=f"**操作失败：** {error_text(exc)}"
        self.start_button.disabled=self.jobs.running
        self.cancel_button.disabled=not self.jobs.running
        self.download_button.disabled=self.jobs.running or self.jobs.directory is None or not (self.jobs.directory/"result.json").is_file()
        if time.monotonic()-self._last_poll<.25:return
        self._last_poll=time.monotonic()
        state=self.jobs.poll()
        if not state:return
        path=state["directory"]
        if state["running"]:
            phase=state["progress"].get("phase","准备实验快照")
            if state['progress'].get('total'):
                phase+=f" · {state['progress'].get('done',0)}/{state['progress']['total']}"
            stages=", ".join(f"{display_text(n.split('/')[-1])}：{display_text(s)}" for n,s in state["pipeline_stages"].items())
            self.status.content=f"运行中 · {state['seconds']:.1f}s · {phase}\n\n{stages}\n\n`{path}`"
        elif self._displayed!=path:
            self._displayed=path;self._show_result(state)

    def _show_result(self,state):
        self.truth_preview.visible=False
        result=state["result"] or {}
        if result.get('algorithm')==MULTITILT:
            self._show_multitilt(state,result)
            return
        status=result.get("status","failed")
        title={"complete":"计算完成（不等于物理标定验收通过）","failed":"标定失败","cancelled":"已取消","rejected":"未通过质量检查"}.get(status,display_text(status))
        self.status.content=f"**{title}** · {display_text(result.get('algorithm','—'))} · 场景版本 {result.get('scene_revision','—')}\n\n`{state['directory']}`"
        lines=[]
        if result.get("error"):lines.append(f"错误：{error_text(result['error'])}")
        if result.get("input_notice"):
            lines.append('使用当前场景采集图片。' if result.get('input_scope')=='current_scene_snapshot'
                         else '使用配置文件指定的数据，不代表当前场景。')
        for r in result.get('ground_truth_comparison',[]):
            row=next((x for x in result.get('summary',[]) if x['kind']=='intrinsics' and x['camera']==r['camera'] and x['model']==r['model']),{})
            if row.get('focal_relative_percent') is not None:
                lines.append(f"{r['camera']} 内参观测最优初值的焦距相对真值误差：{row['focal_relative_percent']:.3f}%（不消除退化）。")
            e=r.get('selected_by_observations')
            if e:lines.append(f"{r['camera']} 观测选择的外参候选：真值平移误差 {1000*e['translation_error_m']:.3f} mm，"
                f"旋转误差 {e['rotation_error_deg']:.4f}°（保留原歧义状态，未按真值挑选候选）。")
        lines.extend(["","| 阶段／相机 | 状态 | 均方根误差（像素） |","| --- | --- | ---: |"])
        for row in result.get("summary",[]):
            rms=row.get("rms_px");rms=f"{rms:.5f}" if isinstance(rms,(int,float)) else "—"
            lines.append(f"| {display_text(row['kind'])}/{display_text(row['model'])}/{camera_name(row['camera'])} | {display_text(row['status'])} | {rms} |")
        for name,row in result.get("results",{}).items():
            e=row["extrinsics_error"]
            lines.append(f"\n{camera_name(name)}：内参 K={row['K_estimated']}；畸变 D={row['D_estimated']}\n\n"
                         f"平移均方根误差 {1000*e['translation_rmse_m']:.3f} mm；旋转均方根误差 {e['rotation_rmse_deg']:.4f}°")
        for row in result.get("summary",[]):
            if row.get("K"):lines.append(f"\n{display_text(row['model'])}／{camera_name(row['camera'])}：内参 K={row['K']}；畸变 D={row['D']}")
            if row.get("kind")=="extrinsics":
                lines.append(f"\n{camera_name(row['camera'])}：{row['candidate_count']} 个条件外参候选，完整矩阵见导出结果；存在候选不代表唯一解。")
            if row.get("kind")=="bev" and row.get("coverage") is not None:
                lines.append(f"\n鸟瞰图 {row['model']} 覆盖率 {100*row['coverage']:.2f}%（不是精度指标）")
        lines.append("\n完整参数、指标和日志已保存至实验目录。估计参数不会自动应用到当前相机。")
        self.results.content="\n".join(lines)
        images=sorted((Path(state["directory"])/"pipeline/bev").glob("*/comparison.png"))
        if images:
            bgr=cv2.imread(str(images[0]))
            if bgr is not None:self.preview.image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);self.preview.visible=True

    def _show_multitilt(self,state,result):
        title=('已取消' if result.get('status')=='cancelled' else
            '计算失败' if result.get('status')=='failed' else
            '内外参数值验收通过（当前合成场景）' if result.get('calibration_accepted') else
            '内参通过，外参未通过' if result.get('intrinsics_accepted') and result.get('scope')=='intrinsics_and_extrinsics' else
            '内参数值检查通过' if result.get('intrinsics_accepted') else '内参未通过或计算失败')
        self.status.content=f"**{title}** · 多倾角鱼眼标定\n\n`{state['directory']}`"
        lines=['结果不会自动写回当前相机。']
        if result.get('error'):lines.append(error_text(result['error']))
        lines += ['', '| 相机 | 状态 | 训练均方根误差（像素） | 留出最大误差（像素） |', '| --- | --- | ---: | ---: |']
        for name,row in result.get('results',{}).items():
            lines.append(f"| {camera_name(name)} | {display_text(row['status'])} | {row['rms_px']:.4f} | {row['heldout_max_frame_rmse_px']:.4f} |")
            lines.append(f"\n{camera_name(name)}：内参 K={row['K']}；畸变 D={row['D']}")
            lines.extend(f"\n{camera_name(name)}：{error_text(reason)}" for reason in row['rejection_reasons'])
        if result.get('extrinsics'):
            lines += ['', '| 外参相机 | 状态 | 均方根误差（像素） | 留出误差（像素） | 编码板数 |', '| --- | --- | ---: | ---: | ---: |']
            for name,row in result['extrinsics'].items():
                rms=f"{row['rms_px']:.4f}" if 'rms_px' in row else '—'
                held=f"{row['heldout_rmse_px']:.4f}" if 'heldout_rmse_px' in row else '—'
                lines.append(f"| {camera_name(name)} | {display_text(row['status'])} | {rms} | {held} | {len(row.get('marker_ids',[]))} |")
                lines.extend(f"\n{camera_name(name)}外参：{error_text(reason)}" for reason in row.get('rejection_reasons',[]))
            lines.append('\n外参将相机光学坐标变换到场景世界坐标。')
        lines.append('\n留出图片不参与内参拟合；位姿拟合与评分使用不同角点。完整记录见实验目录。')
        bev=result.get('bev',{})
        if bev:
            lines.append(f"\n鸟瞰图：{display_text(bev.get('status'))}。" +
                         (error_text(bev['reason']) if bev.get('reason') else '地面假设会使立体物体拉伸，覆盖率不代表精度。'))
            if 'coverage_fraction' in bev:
                lines.append(f"\n鸟瞰覆盖率 {100*bev['coverage_fraction']:.2f}%（不是精度指标），图片：{bev['image']}")
        self.results.content='\n'.join(lines);self.preview.visible=False;self.truth_preview.visible=False
        if bev.get('status')=='GENERATED':
            bgr=cv2.imread(str(Path(state['directory'])/bev['image']))
            if bgr is not None:
                self.preview.image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                self.preview.visible=True
            if bev.get('truth_image'):
                truth_bgr=cv2.imread(str(Path(state['directory'])/bev['truth_image']))
                if truth_bgr is not None:
                    self.truth_preview.image=cv2.cvtColor(truth_bgr,cv2.COLOR_BGR2RGB)
                    self.truth_preview.visible=True

    def close(self):self.jobs.close()
