from copy import deepcopy
import json
import cv2
import numpy as np
import pytest
from scene_editor.calibrated_bev import generate
from scene_editor.scene_pipeline import digest


def test_bev_uses_saved_pixels_and_estimated_parameters_with_hash_check(tmp_path):
    from test_scene_pipeline import fixture_scene
    scene=fixture_scene(tmp_path)
    template=deepcopy(scene.cameras['front'])
    names=['front','rear','left','right']
    for name in names:
        c=deepcopy(template);c.name=name;c.model='fisheye';c.D=[0.]*4
        c.parent='world';c.pose=[0.,0.,2.,0.,np.pi/2,0.]
        c.K=[80.,0.,c.width/2,0.,80.,c.height/2,0.,0.,1.]
        scene.cameras[name]=c
    scene.bev.width=32;scene.bev.height=32
    (tmp_path/'request.json').write_text(json.dumps({'scene_snapshot':scene.to_dict()}))
    lens={};external={}
    for i,name in enumerate(names):
        c=scene.cameras[name]
        pixels=np.full((c.height,c.width,3),40+i*30,np.uint8)
        pixels[:,:,0]=np.arange(c.width,dtype=np.uint8)[None,:]
        path=tmp_path/f'{name}.png';cv2.imwrite(str(path),pixels)
        lens[name]={'intrinsics_accepted':True,'K':c.matrix.tolist(),'D':c.D}
        external[name]={'extrinsics_accepted':True,'image':path.name,'image_sha256':digest(path),
            'T_world_from_camera_optical':scene.camera_optical_pose(name).tolist()}
    result={'calibration_accepted':True,'results':lens,'extrinsics':external}
    metrics=generate(tmp_path,result)
    assert metrics['status']=='GENERATED'
    assert metrics['comparison_order']==['truth','calibrated']
    truth=cv2.imread(str(tmp_path/metrics['truth_image']))
    assert cv2.imread(str(tmp_path/metrics['comparison_image'])).shape==(32,64,3)
    first=cv2.imread(str(tmp_path/'bev/stitched.png'))
    assert first.shape==(32,32,3)
    changed=deepcopy(result)
    for name in names:
        changed['extrinsics'][name]['T_world_from_camera_optical'][0][3]+=2.
    generate(tmp_path,changed,tmp_path/'changed')
    assert not np.array_equal(first,cv2.imread(str(tmp_path/'changed/stitched.png')))
    np.testing.assert_array_equal(truth,cv2.imread(str(tmp_path/'changed/truth_stitched.png')))
    with pytest.raises(FileExistsError):generate(tmp_path,result)
    (tmp_path/'front.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='哈希'):generate(tmp_path,result,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError,match='全部通过'):generate(tmp_path,{'calibration_accepted':False})
