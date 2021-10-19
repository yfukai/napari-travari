"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Napari Travari."""
    zarr_path = path.join(base_dir, "image_total_aligned_small2.zarr")
    #%%
    logger.info("program started")

    #%%
    zarr_file = zarr.open(zarr_path, "r")

    image = da.from_zarr(zarr_file["image"]).persist()
    chunks = zarr_file["image"].chunks

    segment_columns = ["segment_id","bbox_y0","bbox_y1","bbox_x0","bbox_x1"]
    division_columns=["segment_id_parent","frame_child1","label_child1","frame_child2","label_child2"]

    if not read_travari:
        mask = da.from_zarr(zarr_file["mask"])[:, np.newaxis, ...]
        mask[mask == -1] = 0 # mask.max().compute() + 1
        df_segments2_buf=path.join(base_dir, "df_segments2_updated.csv")
        df_divisions2_buf=path.join(base_dir, "df_divisions2.csv")

        finalized_segment_ids=set()
        candidate_segment_ids=set()
    else:
        travari_zarr_path=zarr_path.replace(".zarr","_travari.zarr")
        travari_zarr_file = zarr.open(travari_zarr_path, "r")
        mask_ds=travari_zarr_file["mask"]
        mask = da.from_zarr(mask_ds).persist()
        df_segments2_buf=io.StringIO(mask_ds.attrs["df_segments"].replace("\\n","\n"))
        df_divisions2_buf=io.StringIO(mask_ds.attrs["df_divisions"].replace("\\n","\n"))
        finalized_segment_ids=set(mask_ds.attrs["finalized_segment_ids"])
        candidate_segment_ids=set(mask_ds.attrs["candidate_segment_ids"])
    #    candidate_segment_ids=set()

    sizeT = mask.shape[0]
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

    df_segments2=df_segments2.astype(
        dict(zip(segment_columns,[pd.Int64Dtype()]+[pd.Int32Dtype()]*4))
    )
    df_divisions2=df_divisions2.astype(pd.Int64Dtype())

    #segment_labels = df_segments2.xs(
    #    0, level="frame", drop_level=False
    #).index.get_level_values("label")
    #labels = [0] + sorted(list(set(segment_labels.values)))

    new_label_value = df_segments2.index.get_level_values("label").max() + 1
    assert not np.any(mask ==new_label_value)
    new_segment_id = df_segments2["segment_id"].max() + 1

    #%%
    # TODO set value of label_layer to zero at timeframes not in target_Ts 

    #%%
    _df_segments=df_segments2.copy()
    _df_divisions=df_divisions2.copy()
    _new_segment_id=new_segment_id

    # TODO assert all time steps are in the target_Ts

    travari_viewer=


if __name__ == "__main__":
    main(prog_name="napari-travari")  # pragma: no cover