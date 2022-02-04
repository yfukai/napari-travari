import numpy as np

NOSEL_VALUE = np.iinfo(np.uint32).max
NEW_LABEL_VALUE = NOSEL_VALUE - 1

DF_SEGMENTS_COLUMNS = ["segment_id", "bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1"]
DF_DIVISIONS_COLUMNS = [
    "segment_id_parent",
    "frame_child1",
    "label_child1",
    "frame_child2",
    "label_child2",
]
