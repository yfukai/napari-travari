#%%
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

from ._consts import LOGGING_PATH
from ._logging import logger
from ._viewer import TravaliViewer

DF_SEGMENTS_COLUMNS = ["segment_id", "bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1"]
DF_DIVISIONS_COLUMNS = [
    "segment_id_parent",
    "frame_child1",
    "label_child1",
    "frame_child2",
    "label_child2",
]
LOGGING_PATH = ".travali/log.txt"


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("label_path", type=click.Path(exists=True))
@click.option("--log_directory", "-l", type=click.Path(), default=LOGGING_PATH)
@click.option("--persist", "-p", is_flag=True, default=False)
@click.version_option()
def main(image_path, label_path, log_directory, persist) -> None:
    """Napari Travali."""

    log_path = path.join(path.expanduser("~"), log_directory)
    if not path.exists(path.dirname(log_path)):
        os.makedirs(path.dirname(log_path))
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logger.info("program started")

    zarr_file = zarr.open(image_path, "r")
    image = da.from_zarr(zarr_file["image"])
    if persist:
        image = image.persist()
    data_chunks = zarr_file["image"].chunks

    if label_path is None:
        label_path = image_path.replace(".zarr", "_travali.zarr")

    label_zarr_file = zarr.open(label_path, "r")
    label_ds = label_zarr_file["label"]
    label = da.from_zarr(label_ds)  # .persist()
    if persist:
        label = label.persist()

    df_segments = pd.DataFrame(
        label_zarr_file["df_segments"], columns=DF_SEGMENTS_COLUMNS, index=0
    )
    df_divisions = pd.DataFrame(
        label_zarr_file["df_segments"], columns=DF_DIVISIONS_COLUMNS, index=0
    )

    finalized_segment_ids = set(label_zarr_file.attrs["finalized_segment_ids"])
    candidate_segment_ids = set(label_zarr_file.attrs["candidate_segment_ids"])
    target_Ts = sorted(label_zarr_file.attrs["target_Ts"])
    assert all(np.array(target_Ts) < label.shape[0])

    new_label_value = df_segments.index.get_level_values("label").max() + 1
    new_segment_id = df_segments["segment_id"].max() + 1

    #### only extract information in target_Ts ####
    logger.info("extracting info")

    label[np.setdiff1d(np.arange(label.shape[0]), target_Ts)] = 0
    df_segments = df_segments[
        df_segments.index.get_level_values("frame").isin(target_Ts)
    ].copy()

    def find_alternative_child(frame, label):
        segment_id = df_segments.loc[(frame, label)]["segment_id"]
        df_matched = df_segments[df_segments["segment_id"] == segment_id]
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

    assert all(df_segments.index.get_level_values("frame").isin(target_Ts))
    for i in [1, 2]:
        assert all(df_divisions[f"frame_child{i}"].isin(target_Ts))

    #### run the viewer ####

    logger.info("running viewer")
    _ = TravaliViewer(
        image,
        label,
        target_Ts,
        df_segments,
        df_divisions,
        label_path,
        data_chunks,
        new_segment_id,
        new_label_value,
        finalized_segment_ids,
        candidate_segment_ids,
    )
    napari.run()


if __name__ == "__main__":
    main(prog_name="napari-travali")  # pragma: no cover
