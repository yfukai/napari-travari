# %%
"""Command-line interface."""
import io
import logging
import os
from os import path

import click
import dask.array as da
import napari
import numpy as np
import pandas as pd
import zarr
from _settings._consts import DF_DIVISIONS_COLUMNS
from _settings._consts import DF_TRACKS_COLUMNS
from _settings._consts import LOGGING_PATH
from _settings._transitions import transitions
from _utils._logging import log_error
from _utils._logging import logger
from transitions import Machine

from ._viewer_model import ViewerModel
from ._viewer_model import ViewerState


LOGGING_PATH = ".travali/log.txt"


@click.command()
@click.argument("zarr_path", type=click.Path(exists=True))
@click.argument("label_dataset_name", type=str, default="original")
@click.option("--log_directory", "-l", type=click.Path(), default=LOGGING_PATH)
@click.option("--persist", "-p", is_flag=True, default=False)
@click.version_option()
def main(zarr_path, label_dataset_name, log_directory, persist) -> None:
    """Napari Travali."""

    log_path = path.join(path.expanduser("~"), log_directory)
    if not path.exists(path.dirname(log_path)):
        os.makedirs(path.dirname(log_path))
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logger.info("program started")

    zarr_file = zarr.open(zarr_path, "r")
    image = da.from_zarr(zarr_file["image"])
    label_group = zarr_file["labels"]

    travali_label_dataset_name = label_dataset_name + ".travali"
    if travali_label_dataset_name in label_group.keys():
        label_dataset_name = travali_label_dataset_name
    label_ds = label_group[label_dataset_name]
    label = da.from_zarr(label_ds)
    if persist:
        image = image.persist()
        label = label.persist()
    label = label[:, np.newaxis, :, :, :]
    data_chunks = zarr_file["image"].chunks

    segments_ds = zarr_file["df_tracks"][label_dataset_name]
    df_tracks = pd.DataFrame(segments_ds, columns=DF_TRACKS_COLUMNS).set_index(
        ["frame", "label"]
    )
    df_divisions = pd.DataFrame(
        zarr_file["df_divisions"][label_dataset_name],
        columns=DF_DIVISIONS_COLUMNS,
    )

    finalized_track_ids = set(segments_ds.attrs["finalized_track_ids"])
    candidate_track_ids = set(segments_ds.attrs["candidate_track_ids"])
    termination_annotations = {
        int(k): str(v)
        for k, v in segments_ds.attrs.get("termination_annotations", {}).items()
    }

    target_Ts = sorted(label_ds.attrs["target_Ts"])
    assert all(np.array(target_Ts) < label.shape[0])

    new_label_value = df_tracks.index.get_level_values("label").max() + 1
    new_track_id = df_tracks["track_id"].max() + 1

    #### only extract information in target_Ts ####
    logger.info("extracting info")

    label[np.setdiff1d(np.arange(label.shape[0]), target_Ts)] = 0

    logger.info("organizing dataframes")
    df_tracks = df_tracks[
        df_tracks.index.get_level_values("frame").isin(target_Ts)
    ].copy()

    def find_alternative_child(frame, label):
        track_id = df_tracks.loc[(frame, label)]["track_id"]
        df_matched = df_tracks[df_tracks["track_id"] == track_id]
        if len(df_matched) == 0:
            return (None, None)
        else:
            return df_matched.index.min()  # get the first frame of matched points

    for i in [1, 2]:
        df_divisions[f"frame_child{i}"], df_divisions[f"label_child{i}"] = zip(
            *df_divisions.apply(
                lambda row: find_alternative_child(
                    row[f"frame_child{i}"], row[f"label_child{i}"]
                ),
                axis=1,
            )
        )
        df_divisions[f"frame_child{i}"] = df_divisions[f"frame_child{i}"].astype(
            pd.Int64Dtype()
        )
        df_divisions[f"label_child{i}"] = df_divisions[f"label_child{i}"].astype(
            pd.Int64Dtype()
        )
    df_divisions = df_divisions.dropna()

    assert all(df_tracks.index.get_level_values("frame").isin(target_Ts))
    for i in [1, 2]:
        assert all(df_divisions[f"frame_child{i}"].isin(target_Ts))

    #### run the viewer ####

    logger.info("running viewer")

    target_Ts = sorted(list(map(int, target_Ts)))

    ############### initialize napari viewer ###############
    viewer = napari.Viewer()
    contrast_limits = np.percentile(np.array(image[0]).ravel(), (50, 98))

    viewer.add_image(image, contrast_limits=contrast_limits)

    label_layer = viewer.add_labels(label, name="label", cache=False)
    sel_label_layer = viewer.add_labels(
        da.zeros_like(label, dtype=np.uint8), name="Selected label", cache=False
    )
    sel_label_layer.contour = 3
    redraw_label_layer = viewer.add_labels(
        np.zeros(label.shape[-3:], dtype=np.uint8), name="Drawing", cache=False
    )
    finalized_label_layer = viewer.add_labels(
        da.zeros_like(label, dtype=np.uint8),
        name="Finalized",
        # color ={1:"red"}, not working
        opacity=1.0,
        blending="opaque",
        cache=False,
    )
    finalized_label_layer.contour = 3

    ############### initialize and register state machine model ###############
    viewer_model = ViewerModel(
        self,
        df_tracks,
        df_divisions,
        new_track_id=new_track_id,
        new_label_value=new_label_value,
        finalized_track_ids=finalized_track_ids,
        candidate_track_ids=candidate_track_ids,
        termination_annotations=termination_annotations,
    )
    machine = Machine(
        model=viewer_model,
        states=ViewerState,
        transitions=transitions,
        after_state_change="update_layer_status",
        initial=ViewerState.ALL_LABEL,
        ignore_invalid_triggers=True,  # ignore invalid key presses
    )
    viewer_model.update_layer_status()

    ############### register callbacks ###############
    @log_error
    def track_clicked(layer, event):
        logger.info("Track clicked")
        yield  # important to avoid a potential bug when selecting the daughter
        logger.info("button released")
        global viewer_model
        data_coordinates = layer.world_to_data(event.position)
        cords = np.round(data_coordinates).astype(int)
        val = layer.get_value(data_coordinates)
        if val is None:
            return
        if val != 0:
            msg = f"clicked at {cords}"
            frame = cords[0]
            logger.info("%s %i %s", msg, frame, val)

            row = viewer_model.df_tracks.xs((cords[0], val), level=("frame", "label"))
            if len(row) != 1:
                return
            logger.info(row)
            track_id = row["track_id"].iloc[0]
            viewer_model.track_clicked(frame, val, track_id)

    label_layer.mouse_drag_callbacks.append(track_clicked)

    bind_keys = [
        "Escape",
        "Enter",
        "r",
        "s",
        "d",
        "t",
    ]

    class KeyTyped:
        def __init__(self1, key: str) -> None:
            self1.key = key

        def __call__(self1, _event) -> None:
            logger.info(f"{self1.key} typed")
            getattr(
                viewer_model, f"{self1.key}_typed"
            )()  # call associated function of the model

    for k in bind_keys:
        # register the callback to the viewer
        viewer.bind_key(k, KeyTyped(k), overwrite=True)

    class MoveInTargetTs:
        def __init__(self1, forward: bool):
            self1.forward = forward

        def __call__(self1, _event) -> None:
            # XXX dirty implementation but works
            target_Ts = np.array(target_Ts)
            logger.info(f"moving {self1.forward}")
            iT = viewer.dims.point[0]
            if self1.forward:
                iTs = target_Ts[target_Ts > iT]
                if len(iTs) > 0:
                    viewer.dims.set_point(0, np.min(iTs))
            else:
                iTs = target_Ts[target_Ts < iT]
                if len(iTs) > 0:
                    viewer.dims.set_point(0, np.max(iTs))

    viewer.bind_key("Shift-Right", MoveInTargetTs(True), overwrite=True)
    viewer.bind_key("Shift-Left", MoveInTargetTs(False), overwrite=True)

    @log_error
    def save_typed(_event):
        logger.info("saving validation results...")
        viewer_model.save_results(zarr_path, label_dataset_name, data_chunks, persist)
        logger.info("done.")

    viewer.bind_key("Control-Alt-Shift-S", save_typed, overwrite=True)

    _ = TravaliViewer(
        image,
        label,
        target_Ts,
        df_tracks,
        df_divisions,
        zarr_path,
        label_dataset_name,
        data_chunks,
        new_track_id,
        new_label_value,
        finalized_track_ids,
        candidate_track_ids,
        termination_annotations,
        persist,
    )
    napari.run()


if __name__ == "__main__":
    main(prog_name="napari-travali")  # pragma: no cover
