from enum import Enum

viewer = napari.Viewer()
contrast_limits = np.percentile(np.array(image[0]).ravel(), (50, 98))
viewer.add_image(image, contrast_limits=contrast_limits)
label_layer = viewer.add_labels(mask, name="Mask")
sel_label_layer = viewer.add_labels(
    da.zeros_like(mask, dtype=np.uint8), name="Selected Mask"
)
sel_label_layer.contour=3
redraw_label_layer = viewer.add_labels(
    np.zeros(mask.shape[-3:], dtype=np.uint8), name="Drawing"
)
finalized_label_layer = viewer.add_labels(
    da.zeros_like(mask, dtype=np.uint8), name="Finalized",
   # color ={1:"red"}, not working
    opacity=1.0,blending="opaque"
)
finalized_label_layer.contour=3

layers = [label_layer, sel_label_layer, redraw_label_layer, finalized_label_layer]

class ViewerState(Enum):
    ALL_LABEL = 1
    LABEL_SELECTED = 2
    LABEL_REDRAW = 3
    LABEL_SWITCH = 4
    DAUGHTER_SWITCH = 5
    DAUGHTER_DRAW = 6
    DAUGHTER_CHOOSE_MODE = 7


viewer_state_visibility = {
    ViewerState.ALL_LABEL: [True, False, False, True],
    ViewerState.LABEL_SELECTED: [True, True, False, False],
    ViewerState.LABEL_REDRAW: [False, False, True, False],
    ViewerState.LABEL_SWITCH: [True, False, False, True],
    ViewerState.DAUGHTER_SWITCH: [True, False, False, True],
    ViewerState.DAUGHTER_DRAW: [False, False, True, False],
    ViewerState.DAUGHTER_CHOOSE_MODE: [False, True, False, False],
}

viewer_state_active = {
    ViewerState.ALL_LABEL: "label_layer,
    ViewerState.LABEL_SELECTED: sel_label_layer,
    ViewerState.LABEL_REDRAW: redraw_label_layer,
    ViewerState.LABEL_SWITCH: label_layer,
    ViewerState.DAUGHTER_SWITCH: label_layer,
    ViewerState.DAUGHTER_DRAW: redraw_label_layer,
    ViewerState.DAUGHTER_CHOOSE_MODE: sel_label_layer
}

def set_visible_layers(visibles):
    assert len(visibles) == len(layers)
    for i in range(len(layers)):
        try:
            layers[i].visible = visibles[i]
        except ValueError:
            pass

