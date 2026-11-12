"""Calibration controls shared by standalone and ROS editors."""
from pathlib import Path
from queue import SimpleQueue
import time
import cv2
import numpy as np
from .calibration_jobs import CalibrationJobs,BASELINE,USER_PIPELINE,MULTITILT
from .geometry_ui import BoardGeometryPanel,region_names

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
        with server.gui.add_folder("标定 Calibration",expand_by_default=False):
            self.algorithm=server.gui.add_dropdown("标定算法",options=tuple(ALGORITHMS))
            self.info=server.gui.add_markdown("OpenCV：冻结当前场景和相机参数，生成图片再标定，不修改现有相机。")
            self.algorithm.on_update(lambda _:self.events.put("algorithm"))
            self.input_mode=server.gui.add_dropdown('1.0.0 输入来源',options=tuple(INPUT_MODES))
            self.input_mode.on_update(lambda _:self.events.put('algorithm'))
            self.experiment_id=server.gui.add_text("实验名称（留空自动生成）","")
            self.camera=server.gui.add_dropdown("待标定相机",options=(ALL_CAMERAS,))
            self.board=server.gui.add_dropdown("标定板",options=("(无标定板)",))
            self.views=server.gui.add_number("每相机采样数",40,min=8,max=500,step=1)
            self.seed=server.gui.add_number("随机 seed",20261020,min=0,max=2**32-1,step=1)
            self.supersample=server.gui.add_number("超采样倍数",2,min=1,max=4,step=1)
            self.backend=server.gui.add_dropdown("Gaussian 板渲染后端",options=("cpu","gsplat"),initial_value=backend)
            self.geometry_ok=server.gui.add_checkbox("已核实 Gaussian 为完整矩形棋盘且格长/朝向正确",initial_value=False)
            self.intrinsic_board=server.gui.add_dropdown('内参运动小板（完整矩形）',options=('(无)',))
            self.extrinsic_board=server.gui.add_dropdown('外参固定大板（region/条带）',options=('(无)',))
            self.extrinsic_board.on_update(lambda _:self.events.put('regions'))
            self.holdout=server.gui.add_number('每相机留出图片数',2,min=1,max=30,step=1)
            self.spread_x=server.gui.add_number('小板局部 X 平移半范围（m）',.5,min=0,max=20,step=.05)
            self.spread_y=server.gui.add_number('小板局部 Y 平移半范围（m）',.5,min=0,max=20,step=.05)
            self.yaw_range=server.gui.add_number('小板面内旋转半范围（deg）',30.,min=0,max=180,step=1.)
            self.sampling=server.gui.add_dropdown('平面采样方式',options=('自动：逐相机可见平面','当前小板附近局部 XY 范围'))
            self.mapping_folder=server.gui.add_folder('相机 → 大板 region',expand_by_default=True)
            self.save_settings=server.gui.add_button('保存本场景标定设置')
            self.save_settings.on_click(lambda _:self.events.put('save_settings'))
            self.pipeline_config=server.gui.add_text("1.0.0 配置 JSON（仅高级模式）",'')
            self.stage=server.gui.add_text("1.0.0 阶段（all 或有序逗号列表）","all")
            self.tilt=server.gui.add_number('小板最大倾角（deg）',40.,min=20.,max=55.,step=1.)
            self.distance_min=server.gui.add_number('最小采集距离（m）',1.,min=.4,max=10.,step=.1)
            self.distance_max=server.gui.add_number('最大采集距离（m）',2.4,min=.4,max=10.,step=.1)
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
            self.results=server.gui.add_markdown("结果会显示在这里；完整参数与日志保存到 experiments/。")
            self.preview=server.gui.add_image(np.zeros((64,128,3),np.uint8),label="标定参数鸟瞰图 / 流程 BEV")
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
            self.info.content='固定相机，自动改变小板倾角、距离和位置；留出图不参与内参拟合。可继续使用已知 ID 和公开三维角点的编码板求外参。自动从资产包查找几何 JSON，也可手工指定；缺失时启动前报错。不会自动修改相机。'
            return
        self.info.content=("冻结点击时的当前 SceneState；运行期间可继续编辑。"
            "结果不自动覆盖相机参数。Gaussian 环车条带不能当完整棋盘。" if baseline else
            '冻结当前场景，使用原相机和 gsplat 直出 PNG；不是 Graphdeco 六面采集。'
            '相机不动。自动模式可逐相机布置小板；每路采集固定板平面，仅面内平移/旋转。内参采集后移除内参板，其他板不变；'
            '高斯板缺少真实角点定义时会拒绝，先在公开几何面板填写。真值不传求解器。' if current else
            "执行你提供的原版 1.0.0 流程，数据来源由 JSON 决定；演示模式使用独立 CPU 示例，"
            "不是当前场景重新采集。当前场景快照仅作为实验上下文。"
            "退化、局部解和角点歧义将原样保留；流程完成不等于标定验收通过。")

    def settings(self):
        if ALGORITHMS[self.algorithm.value]==MULTITILT:
            return {'algorithm':MULTITILT,'experiment_id':self.experiment_id.value,
                'cameras':None if self.camera.value==ALL_CAMERAS else [self.camera.value],
                'intrinsic_board':self.intrinsic_board.value,'views':int(self.views.value),
                'holdout':int(self.holdout.value),'seed':int(self.seed.value),
                'supersample':int(self.supersample.value),'backend':self.backend.value,
                'tilt_deg':float(self.tilt.value),'distance_min':float(self.distance_min.value),
                'distance_max':float(self.distance_max.value),
                'run_extrinsics':self.run_extrinsics.value,
                'marker_geometry_file':self.marker_geometry_file.value.strip()}
        return {"algorithm":ALGORITHMS[self.algorithm.value],"experiment_id":self.experiment_id.value,
            "cameras":None if self.camera.value==ALL_CAMERAS else [self.camera.value],
            "board_id":self.board.value,"views":int(self.views.value),"seed":int(self.seed.value),
            "supersample":int(self.supersample.value),"backend":self.backend.value,
            "confirm_board_geometry":self.geometry_ok.value,"pipeline_config":self.pipeline_config.value,
            "stage":self.stage.value.strip() or 'all','input_mode':INPUT_MODES[self.input_mode.value],
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
                self.region_controls[n]=self.server.gui.add_dropdown(f'{n} region',options=options,
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
        if saved.get('backend') in ('cpu','gsplat'):self.backend.value=saved['backend']
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
            self.camera.options=(ALL_CAMERAS,*ids[0])
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
                    self.status.content='标定角色/采样/映射已写入 SceneState；请 Save Scene 保存 YAML。'
                elif action=="start":
                    settings=self.settings()
                    directory=self.jobs.start(self.store,settings)
                    self.status.content=f"后台标定已启动：`{directory}`"
                    self.results.content="正在运行，不会覆盖当前相机参数。"
                    self.preview.visible=False;self.truth_preview.visible=False;self._displayed=None
                elif action=="cancel":self.jobs.cancel()
            except (ValueError,OSError,KeyError,RuntimeError,TypeError) as exc:
                self.status.content=f"**无法开始标定：** {exc}"
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
            stages=", ".join(f"{n}: {s}" for n,s in state["pipeline_stages"].items())
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
        title={"complete":"计算完成（不等于物理标定验收通过）","failed":"标定失败","cancelled":"已取消"}.get(status,status)
        self.status.content=f"**{title}** · {result.get('algorithm','—')} · scene revision {result.get('scene_revision','—')}\n\n`{state['directory']}`"
        lines=[]
        if result.get("error"):lines.append(f"错误：{result['error']}")
        if result.get("input_notice"):lines.append(result["input_notice"])
        for r in result.get('ground_truth_comparison',[]):
            row=next((x for x in result.get('summary',[]) if x['kind']=='intrinsics' and x['camera']==r['camera'] and x['model']==r['model']),{})
            if row.get('focal_relative_percent') is not None:
                lines.append(f"{r['camera']} 内参观测最优初值的焦距相对 GT 误差：{row['focal_relative_percent']:.3f}%（不消除退化）。")
            e=r.get('selected_by_observations')
            if e:lines.append(f"{r['camera']} 观测选择的外参候选：GT 平移误差 {1000*e['translation_error_m']:.3f} mm，"
                f"旋转误差 {e['rotation_error_deg']:.4f}°（保留原歧义状态，未按 GT 挑选候选）。")
        lines.extend(["","| 阶段/相机 | 状态 | RMS (px) |","| --- | --- | ---: |"])
        for row in result.get("summary",[]):
            rms=row.get("rms_px");rms=f"{rms:.5f}" if isinstance(rms,(int,float)) else "—"
            lines.append(f"| {row['kind']}/{row['model']}/{row['camera']} | {row['status']} | {rms} |")
        for name,row in result.get("results",{}).items():
            e=row["extrinsics_error"]
            lines.append(f"\n{name}: K={row['K_estimated']}; D={row['D_estimated']}\n\n"
                         f"平移 RMSE {1000*e['translation_rmse_m']:.3f} mm；旋转 RMSE {e['rotation_rmse_deg']:.4f}°")
        for row in result.get("summary",[]):
            if row.get("K"):lines.append(f"\n{row['model']}/{row['camera']}: K={row['K']}; D={row['D']}")
            if row.get("kind")=="extrinsics":
                lines.append(f"\n{row['model']}/{row['camera']}: {row['candidate_count']} 个条件外参候选。")
                p=row.get("pose_candidate_preview",{})
                if p:
                    lines.append(f"\n全部对应点 RMS 最小的候选（不是唯一解）：R_w2c={p['R_w2c']}; tvec={p['tvec']}")
            if row.get("kind")=="bev" and row.get("coverage") is not None:
                lines.append(f"\nBEV {row['model']} 覆盖率 {100*row['coverage']:.2f}%（不是精度指标）")
        lines.append("\n完整结果：result.json；汇总：metrics.csv；日志：logs/calibration.log。不会自动应用估计参数。")
        self.results.content="\n".join(lines)
        images=sorted((Path(state["directory"])/"pipeline/bev").glob("*/comparison.png"))
        if images:
            bgr=cv2.imread(str(images[0]))
            if bgr is not None:self.preview.image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);self.preview.visible=True

    def _show_multitilt(self,state,result):
        title=('内外参数值验收通过（当前合成场景）' if result.get('calibration_accepted') else
            '内参通过，外参未通过' if result.get('intrinsics_accepted') and result.get('scope')=='intrinsics_and_extrinsics' else
            '内参数值检查通过' if result.get('intrinsics_accepted') else '内参未通过或计算失败')
        self.status.content=f"**{title}** · 多倾角鱼眼标定\n\n`{state['directory']}`"
        lines=[result.get('input_notice','仅内参，不自动修改相机，不生成外参/BEV。')]
        if result.get('error'):lines.append(result['error'])
        lines += ['', '| 相机 | 状态 | 训练 RMS | 留出最大 RMS |', '| --- | --- | ---: | ---: |']
        for name,row in result.get('results',{}).items():
            lines.append(f"| {name} | {row['status']} | {row['rms_px']:.4f} | {row['heldout_max_frame_rmse_px']:.4f} |")
            lines.append(f"\n{name}: K={row['K']}; D={row['D']}")
            lines.extend(f"\n{name}: {reason}" for reason in row['rejection_reasons'])
        if result.get('extrinsics'):
            lines += ['', '| 外参相机 | 状态 | RMS (px) | 留出 RMS (px) | 编码板数 |', '| --- | --- | ---: | ---: | ---: |']
            for name,row in result['extrinsics'].items():
                rms=f"{row['rms_px']:.4f}" if 'rms_px' in row else '—'
                held=f"{row['heldout_rmse_px']:.4f}" if 'heldout_rmse_px' in row else '—'
                lines.append(f"| {name} | {row['status']} | {rms} | {held} | {len(row.get('marker_ids',[]))} |")
                lines.extend(f"\n{name} 外参: {reason}" for reason in row.get('rejection_reasons',[]))
            lines.append('\n外参见 extrinsics/；坐标为 scene_world，T_world_from_camera_optical 将光学坐标变换到世界坐标。')
        lines.append('\n完整观测 solver_inputs/；采集记录 capture/；结果 multitilt/。留出帧只拟合位姿，使用未参与位姿拟合的角点评分。')
        bev=result.get('bev',{})
        if bev:
            lines.append(f"\n鸟瞰图：{bev.get('status')}；{bev.get('reason',bev.get('notice',''))}")
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
