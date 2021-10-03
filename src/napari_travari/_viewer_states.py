from enum import Enum

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
    ViewerState.LABEL_SELECTED: [False, True, False, False],
    ViewerState.LABEL_REDRAW: [False, False, True, False],
    ViewerState.LABEL_SWITCH: [True, False, False, True],
    ViewerState.DAUGHTER_SWITCH: [True, False, False, True],
    ViewerState.DAUGHTER_DRAW: [False, False, True, False],
    ViewerState.DAUGHTER_CHOOSE_MODE: [False, True, False, False],
}

viewer_state_active = {
    ViewerState.ALL_LABEL: label_layer,
    ViewerState.LABEL_SELECTED: sel_label_layer,
    ViewerState.LABEL_REDRAW: redraw_label_layer,
    ViewerState.LABEL_SWITCH: label_layer,
    ViewerState.DAUGHTER_SWITCH: label_layer,
    ViewerState.DAUGHTER_DRAW: redraw_label_layer,
    ViewerState.DAUGHTER_CHOOSE_MODE: sel_label_layer
}

