from qtpy.QtWidgets import QMessageBox, QInputDialog
from _logging import get_logger

logger = get_logger()

def choose_by_mbox(_viewer, choices, message):
    msgBox = QMessageBox(_viewer.window.qt_viewer)
    msgBox.setText(message)
    msgBox.setIcon(QMessageBox.Question)
    buttons = []
    for choice in choices:
        button = msgBox.addButton(choice, QMessageBox.ActionRole)
        buttons.append(button)
    cancelled = msgBox.addButton(QMessageBox.Cancel)
    logger.info("messagebox selected")
    _ = msgBox.exec_()
    clicked_button = msgBox.clickedButton()

    if clicked_button == cancelled:
        return False
    try:
        return choices[buttons.index(clicked_button)]
    except ValueError:
        return None

def choose_direction_by_mbox(_viewer):
    return choose_by_mbox(
        _viewer,
        ["forward", "backward"],
        "Select the time direction of the new track",
    )

def choose_division_by_mbox(_viewer):
    return choose_by_mbox(
        _viewer,
        ["select", "draw"],
        "Select or draw the daughter?",
    )

def ask_draw_label(_viewer):
    return choose_by_mbox(
        _viewer,
        ["modify", "new"],
        "Modify label, or create new?"
    )
    
def ask_ok_or_not(_viewer,message):
    dialogue_result = choose_by_mbox(
        _viewer,
        ["Ok"],
        message
    )
    return dialogue_result=="Ok"

def get_annotation_of_track_end(_viewer):
    return QInputDialog.getText(
        _viewer.window.qt_viewer, "Input dialog", "Annotate the track end:"
    )
