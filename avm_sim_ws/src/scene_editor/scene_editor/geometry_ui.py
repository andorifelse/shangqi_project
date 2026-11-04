"""Public target geometry authoring, without asking users to write pipeline JSON."""
from pathlib import Path
from queue import SimpleQueue
from copy import deepcopy
import json
import numpy as np
from scene_manager.board_geometry import grid_geometry,board_geometry,validate_geometry


class BoardGeometryPanel:
    def __init__(self,server,store):
        self.server,self.store=server,store;self.events=SimpleQueue();self._ids=None
        with server.gui.add_folder('标定板公开几何 / Regions',expand_by_default=False):
            server.gui.add_markdown('这里只填写公开实物几何，不填写相机 K/D/外参。'
                '虚拟板尺寸已知；高斯板必须核实真实格长与角点坐标，不能从 PLY 包围盒冒认。')
            self.board=server.gui.add_dropdown('编辑几何的板',options=('(无)',))
            self.region=server.gui.add_text('region 名称','region_front')
            self.columns=server.gui.add_number('区域内角点列数',7,min=3,max=100,step=1)
            self.rows=server.gui.add_number('区域内角点行数',5,min=3,max=100,step=1)
            self.pitch=server.gui.add_number('已核实格长（坐标单位）',.15,min=.000001,step=.01)
            self.frame=server.gui.add_dropdown('角点坐标系',options=('board','asset'))
            self.origin=[server.gui.add_number(f'第0行0列 {axis}',0.,step=.01) for axis in ('X','Y','Z')]
            self.yaw=server.gui.add_number('网格面内角度（deg）',0.,step=1.)
            self.source=server.gui.add_text('尺寸/角点来源（测量/CAD等）','')
            self.verified=server.gui.add_checkbox('我已核实该区域实际格长/坐标与 PLY 对应',initial_value=False)
            server.gui.add_markdown('生成区域在所选坐标系 XY 面；origin 为第0行0列。'
                '原始 XZ 板请先用 asset_pose 对齐到 board XY，或导入任意朝向的显式角点。'
                '不要把环车中央孔两侧角点连成一个矩形 region。')
            server.gui.add_button('保存/替换这个 region').on_click(lambda _:self.events.put('grid'))
            self.file=server.gui.add_text('公开几何 JSON 路径（可选高级导入）','')
            server.gui.add_button('导入公开角点定义').on_click(lambda _:self.events.put('import'))
            self.status=server.gui.add_markdown('几何随 Save Scene / Ground Truth 保存。')
        self.tick()

    def tick(self):
        _,scene=self.store.snapshot();ids=tuple(scene.boards)
        if ids!=self._ids:self.board.options=ids or ('(无)',);self._ids=ids
        while not self.events.empty():
            action=self.events.get()
            try:
                if self.board.value not in scene.boards:raise ValueError('请先添加或选择标定板')
                b=scene.boards[self.board.value]
                if action=='grid':
                    if not self.verified.value or not self.source.value.strip():
                        raise ValueError('请核实真实几何并填写来源；未确认的默认数值不能作为实物角点')
                    doc=grid_geometry(self.region.value.strip(),int(self.columns.value),int(self.rows.value),
                        float(self.pitch.value),[h.value for h in self.origin],np.deg2rad(self.yaw.value),
                        source=self.source.value.strip(),frame=self.frame.value)
                    old=deepcopy(b.calibration_geometry)
                    if old and old.get('frame')!=doc['frame']:
                        raise ValueError('已有区域坐标系不同；请统一 board/asset 后再合并')
                    if old:
                        doc['corners']=[r for r in old['corners'] if r['region']!=self.region.value.strip()]+doc['corners']
                        doc['source']=old['source']+'; '+doc['source']
                else:
                    doc=json.loads(Path(self.file.value).expanduser().read_text(encoding='utf-8'))
                validate_geometry(doc)
                self.store.command({'action':'properties','id':b.id,'properties':{'calibration_geometry':doc}})
                self.status.content=f"已保存 {b.id}：{len(doc['corners'])} 个角点；"+', '.join(sorted({r['region'] for r in doc['corners']}))
            except (ValueError,KeyError,OSError,TypeError) as exc:self.status.content=f'**几何未保存：** {exc}'


def region_names(board):
    try:return sorted({r['region'] for r in board_geometry(board)['corners']})
    except ValueError:return []
