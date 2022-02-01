#%%
"""Command-line interface."""
import io
import logging
import os
import sys
from os import path

import click
import dask.array as da
import napari
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

from ._consts import LOGGING_PATH
from ._logging import logger
from ._viewer import TravaliViewer


#%%
@click.command()
@click.version_option()
def main() -> None:
    """Napari Travali."""
    read_travali = True
    base_dir = (
        "/home/fukai/projects/microscope_analysis/old/LSM800_2021-03-04-timelapse_old"
    )
    #    base_dir = "/home/fukai/microscope_analysis/old/LSM800_2021-03-04-timelapse_old"

    log_path = path.join(path.expanduser("~"), LOGGING_PATH)
    if not path.exists(path.dirname(log_path)):
        os.makedirs(path.dirname(log_path))
    _root_logger = logging.getLogger()
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logger.info("program started")

    zarr_path = path.join(base_dir, "image_total_aligned_small2.zarr")

    #%%
    zarr_file = zarr.open(zarr_path, "r")

    image = da.from_zarr(zarr_file["image"])  # .persist()
    data_chunks = zarr_file["image"].chunks

    segment_columns = ["segment_id", "bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1"]
    division_columns = [
        "segment_id_parent",
        "frame_child1",
        "label_child1",
        "frame_child2",
        "label_child2",
    ]

    if not read_travali:
        mask = da.from_zarr(zarr_file["mask"])[:, np.newaxis, ...]
        mask[mask == -1] = 0  # mask.max().compute() + 1
        df_segments2_buf = path.join(base_dir, "df_segments2_updated.csv")
        df_divisions2_buf = path.join(base_dir, "df_divisions2.csv")

        finalized_segment_ids = set()
        candidate_segment_ids = set()
    else:
        travali_zarr_path = zarr_path.replace(".zarr", "_travali.zarr")
        travali_zarr_file = zarr.open(travali_zarr_path, "r")
        mask_ds = travali_zarr_file["mask"]
        mask = da.from_zarr(mask_ds)  # .persist()
        df_segments2_buf = io.StringIO(
            mask_ds.attrs["df_segments"].replace("\\n", "\n")
        )
        df_divisions2_buf = io.StringIO(
            mask_ds.attrs["df_divisions"].replace("\\n", "\n")
        )
        finalized_segment_ids = set(mask_ds.attrs["finalized_segment_ids"])
        candidate_segment_ids = set(mask_ds.attrs["candidate_segment_ids"])
        target_Ts = sorted(mask_ds.attrs["target_Ts"])
        #        green_signal=[np.any(im[1,0,:10,:10]>0).compute() for im in tqdm(image)]
        #        target_Ts=list(np.arange(0,mask.shape[0]-3)[green_signal[:-3]])+list(mask.shape[0]-3+np.arange(3))
        assert all(np.array(target_Ts) < mask.shape[0])

    df_segments2 = pd.read_csv(
        df_segments2_buf,
        index_col=["frame", "label"],
        #    dtype = pd.Int64Dtype()
    )[segment_columns]
    df_divisions2 = pd.read_csv(
        df_divisions2_buf,
        index_col=0,
        #    dtype = pd.Int64Dtype()
    )[division_columns]

    df_segments2 = df_segments2.astype(
        dict(zip(segment_columns, [pd.Int64Dtype()] + [pd.Int32Dtype()] * 4))
    )
    df_divisions2 = df_divisions2.astype(pd.Int64Dtype())

    new_label_value = df_segments2.index.get_level_values("label").max() + 1
    # assert not np.any(mask ==new_label_value)
    new_segment_id = df_segments2["segment_id"].max() + 1

    #### only extract information in target_Ts ####
    logger.info("extracting info")

    mask[np.setdiff1d(np.arange(mask.shape[0]), target_Ts)] = 0

    df_segments = df_segments2[
        df_segments2.index.get_level_values("frame").isin(target_Ts)
    ].copy()

    def find_alternative_child(frame, label):
        segment_id = df_segments2.loc[(frame, label)]["segment_id"]
        df_matched = df_segments[df_segments["segment_id"] == segment_id]
        if len(df_matched) == 0:
            return (None, None)
        else:
            return df_matched.index.min()  # get the first frame of matched points

    df_divisions = df_divisions2.copy()
    for i in [1, 2]:
        df_divisions[f"frame_child{i}"], df_divisions[f"label_child{i}"] = zip(
            *df_divisions2.apply(
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
        mask,
        target_Ts,
        df_segments,
        df_divisions,
        zarr_path,
        data_chunks,
        new_segment_id,
        new_label_value,
        finalized_segment_ids,
        candidate_segment_ids,
    )
    napari.run()


if __name__ == "__main__":
    main(prog_name="napari-travali")  # pragma: no cover
# %%
