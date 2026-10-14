import numpy as np
from scene_editor.preview import SensorPreviewBuffer


def test_preview_rejects_old_revision_and_never_mixes_camera_batches():
    buffer = SensorPreviewBuffer()
    names = ("rear", "left", "right", "front")
    old, new = (1, 0), (2, 0)
    buffer.metadata(old, 3, names)
    buffer.metadata(new, 4, names)
    for n in names:
        buffer.image(old, n, np.zeros((2,2,3), dtype=np.uint8))
    assert buffer.ready(4) is None
    assert old not in buffer.batches
    for i, n in enumerate(names[:-1]):
        buffer.image(new, n, np.full((2,2,3), i+1, dtype=np.uint8))
    assert buffer.ready(4) is None
    buffer.image(new, "front", np.full((2,2,3), 4, dtype=np.uint8))
    stamp, images = buffer.ready(4)
    assert stamp == new
    for i, n in enumerate(names):
        assert np.all(images[n] == i+1)
