"""Public target geometry. No camera truth or image-projected IDs belong here."""
from copy import deepcopy
import re
import numpy as np
from .transforms import pose_matrix,transform_points


def validate_geometry(document):
    if not isinstance(document,dict):raise ValueError('标定板几何必须是对象')
    if not document:return
    if document.get('frame') not in ('board','asset'):
        raise ValueError('角点坐标系必须为 board 或 asset')
    if not isinstance(document.get('verified'),bool) or not isinstance(document.get('source'),str):
        raise ValueError('板几何需要 verified 和来源 source')
    rows=document.get('corners')
    if not isinstance(rows,list) or not rows:raise ValueError('板几何缺少公开角点列表')
    identities=set();cells=set()
    for row in rows:
        if not isinstance(row,dict):raise ValueError('每个角点必须包含 ID、region、行列与坐标')
        region=row.get('region');name=row.get('id');r=row.get('grid_row');c=row.get('grid_col')
        if not isinstance(region,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',region):
            raise ValueError('region 必须是英文字母开头的区域名称')
        if not isinstance(name,str) or not name or name in identities:raise ValueError('角点 ID 缺失或重复')
        if any(isinstance(v,bool) or not isinstance(v,int) for v in (r,c)):
            raise ValueError('角点行列必须为整数')
        if (region,r,c) in cells:raise ValueError('同一区域行列角点重复')
        xyz=np.asarray(row.get('xyz_m'),float)
        if xyz.shape!=(3,) or not np.isfinite(xyz).all():raise ValueError('角点 xyz_m 必须是三个有限数值')
        identities.add(name);cells.add((region,r,c))


def grid_geometry(region, columns, rows, pitch, origin=(0,0,0), yaw=0., *, source, frame='board'):
    """Build a user-declared grid: origin is row 0 / column 0, not board center."""
    if (isinstance(rows,bool) or isinstance(columns,bool) or not isinstance(rows,int)
            or not isinstance(columns,int) or not 3<=rows<=100 or not 3<=columns<=100):
        raise ValueError('区域至少需要 3×3 个内角点，最多 100×100')
    if not np.isfinite(pitch) or pitch<=0:raise ValueError('真实格长必须为正数')
    t=pose_matrix([*origin,0.,0.,yaw]);corners=[]
    for r in range(rows):
        for c in range(columns):
            corners.append({'id':f'{region}/r{r:02d}/c{c:02d}','region':region,
                'grid_row':r,'grid_col':c,'xyz_m':transform_points(t,np.array([c*pitch,r*pitch,0.])).tolist()})
    out={'frame':frame,'verified':True,'source':source,'corners':corners}
    validate_geometry(out)
    return out


def board_geometry(board):
    """Analytic boards have known metric geometry; PLYs never inherit a fake grid."""
    document=deepcopy(board.calibration_geometry)
    if not document:
        if board.is_gaussian:
            raise ValueError(f'{board.id}：高斯板未定义已核实的区域/角点。请提供真实格长、角点 ID 和模型局部坐标；不能使用默认 6×8 参数。')
        document=grid_geometry(board.id,board.columns-1,board.rows-1,board.square_size,
            (-board.width/2+board.square_size,-board.height/2+board.square_size,0.),
            source='analytic checkerboard definition')
    validate_geometry(document)
    if not document['verified'] or not document['source'].strip():
        raise ValueError(f'{board.id}：请先核实角点几何并填写尺寸/坐标来源')
    if document['frame']=='asset':
        for row in document['corners']:
            row['xyz_m']=transform_points(pose_matrix(board.asset_pose),np.array(row['xyz_m'])*board.asset_scale).tolist()
        document['frame']='board'
    return document


def regions_of(board):
    doc=board_geometry(board)
    return sorted({r['region'] for r in doc['corners']})
