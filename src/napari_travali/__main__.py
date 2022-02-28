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

from ._consts import DF_DIVISIONS_COLUMNS
from ._consts import DF_SEGMENTS_COLUMNS
from ._consts import LOGGING_PATH
from ._logging import logger
from ._viewer import TravaliViewer


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

    segments_ds = zarr_file["df_segments"][label_dataset_name]
    df_segments = pd.DataFrame(segments_ds, columns=DF_SEGMENTS_COLUMNS).set_index(
        ["frame", "label"]
    )
    df_divisions = pd.DataFrame(
        zarr_file["df_divisions"][label_dataset_name],
        columns=DF_DIVISIONS_COLUMNS,
    )

    finalized_segment_ids = set(segments_ds.attrs["finalized_segment_ids"])
    candidate_segment_ids = set(segments_ds.attrs["candidate_segment_ids"])
    termination_annotations = { int(k):str(v) 
        for k,v in segments_ds.attrs.get("termination_annotations", {}).items()}

    target_Ts = sorted(label_ds.attrs["target_Ts"])
    assert all(np.array(target_Ts) < label.shape[0])

    new_label_value = df_segments.index.get_level_values("label").max() + 1
    new_segment_id = df_segments["segment_id"].max() + 1

    #### only extract information in target_Ts ####
    logger.info("extracting info")

    label[np.setdiff1d(np.arange(label.shape[0]), target_Ts)] = 0

    logger.info("organizing dataframes")
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
        zarr_path,
        label_dataset_name,
        data_chunks,
        new_segment_id,
        new_label_value,
        finalized_segment_ids,
        candidate_segment_ids,
        termination_annotations,
        persist,
    )
    napari.run()


if __name__ == "__main__":
    main(prog_name="napari-travali")  # pragma: no cover
