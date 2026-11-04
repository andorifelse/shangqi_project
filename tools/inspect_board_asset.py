"""Report authoring-grid evidence; never declare inferred PLY units as meters."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from plyfile import PlyData


def inspect(path, subdivisions):
    v=PlyData.read(str(path))['vertex'].data
    xyz=np.column_stack([v[n] for n in ('x','y','z')])
    axes=[np.unique(xyz[:,i]) for i in range(3)]
    planar=sorted(range(3),key=lambda i:len(axes[i]),reverse=True)[:2]
    planar=sorted(planar);thin=next(i for i in range(3) if i not in planar)
    a,b=planar;nx,ny=len(axes[a]),len(axes[b])
    if nx%subdivisions or ny%subdivisions:raise ValueError('采样数不是声明细分数的整数倍，不能推断规则棋盘')
    dx=float(np.median(np.diff(axes[a])));dy=float(np.median(np.diff(axes[b])))
    if not np.allclose(np.diff(axes[a]),dx,rtol=1e-3,atol=1e-6) or not np.allclose(np.diff(axes[b]),dy,rtol=1e-3,atol=1e-6):
        raise ValueError('采样轴不是均匀网格，拒绝推断')
    ix=np.searchsorted(axes[a],xyz[:,a])//subdivisions;iy=np.searchsorted(axes[b],xyz[:,b])//subdivisions
    occupied=np.zeros((ny//subdivisions,nx//subdivisions),bool);occupied[iy,ix]=True
    missing=np.argwhere(~occupied)
    return {'asset':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'gaussians':len(v),'authoring_axes':[('x','y','z')[i] for i in planar],
        'thin_axis':('x','y','z')[thin],'thin_values':axes[thin].tolist(),
        'declared_subdivisions_per_square':subdivisions,'subdivision_source':'asset README K=16; supplied as explicit CLI parameter',
        'inferred_grid_squares':[occupied.shape[1],occupied.shape[0]],
        'inferred_pitch_AUTHORING_UNITS':[dx*subdivisions,dy*subdivisions],
        'present_cells':int(occupied.sum()),'missing_cells':len(missing),
        'missing_cell_bounds_inclusive':None if not len(missing) else {'column':[int(missing[:,1].min()),int(missing[:,1].max())],
            'row':[int(missing[:,0].min()),int(missing[:,0].max())]},
        'is_default_6x8_complete_board':False,'metric_scale_confirmed':False,'corner_ids_confirmed':False,
        'needed_user_evidence':['真实米制格长或模型单位→米尺度','小板资产及完整矩形内角点定义',
            '环车各 region 的角点行列/原点/方向/公开ID','camera→region 人工对应'],
        'caution':'Regular sampling/layout inference is not measured metric geometry or a verified physical corner definition.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ply',type=Path,required=True)
    p.add_argument('--subdivisions',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.subdivisions<=0:p.error('subdivisions must be positive')
    if a.output.exists():p.error('Refusing to overwrite output')
    r=inspect(a.ply,a.subdivisions);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(r,ensure_ascii=False,indent=2))
