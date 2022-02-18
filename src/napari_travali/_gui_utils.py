from qtpy.QtWidgets import QInputDialog
from qtpy.QtWidgets import QMessageBox

from ._logging import logger


def __choose_by_mbox(viewer, choices, message):
    msgBox = QMessageBox(viewer.window.qt_viewer)
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


def choose_direction_by_mbox(viewer):
    return __choose_by_mbox(
        viewer,
        ["forward", "backward"],
        "Select the time direction of the new track",
    )


def choose_division_by_mbox(viewer):
    return __choose_by_mbox(
        viewer,
        ["select", "draw"],
        "Select or draw the daughter?",
    )


def ask_draw_label(viewer):
    return __choose_by_mbox(viewer, ["modify", "new"], "Modify label, or create new?")


def ask_ok_or_not(viewer, message):
    dialogue_result = __choose_by_mbox(viewer, ["Ok"], message)
    return dialogue_result == "Ok"


def get_annotation_of_track_end(viewer,text=""):
    return QInputDialog.getText(
        viewer.window.qt_viewer, "Input dialog", "Annotate the track end:",
        text=text
    )
