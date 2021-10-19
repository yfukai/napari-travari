import napari
import numpy as np
from dask import array as da
from transitions import Machine
from ._viewer_model import ViewerModel, ViewerState
from ._transitions import transitions
from ._logging import log_error, logger

class TravariViewer:
    def __init__(self, image, mask, 
                 new_segment_id = None, 
                 new_label_value = None,
                 finalized_segment_ids = None,
                 candidate_segment_ids = None
                 ):
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

    
        self.viewer_model = ViewerModel(
            new_segment_id=new_segment_id,
            new_label_value=new_label_value,
            finalized_segment_ids=finalized_segment_ids,
            candidate_segment_ids=candidate_segment_ids
            )
        self.machine = Machine(
            model=self.viewer_model,
            states=ViewerState,
            transitions=transitions,
            after_state_change="update_layer_status",
            initial=ViewerState.ALL_LABEL,
            ignore_invalid_triggers=True, # ignore invalid key presses 
        )

        @label_layer.mouse_drag_callbacks.append
        @log_error
        def track_clicked(layer, event):
            logger.info("Track clicked")
            yield # important to avoid a potential bug when selecting the daughter
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
                logger.info("%s %i %s",msg,frame,val)

                row = self.viewer_model.df_segments.xs((cords[0], val), level=("frame", "label"))
                if len(row) != 1:
                    return
                logger.info(row)
                segment_id = row["segment_id"].iloc[0]

                self.viewer_model.track_clicked(frame, val ,segment_id)

        @viewer.bind_key("Escape", overwrite=True)
        def Escape_typed(event):
            logger.info("Escape typed")
            self.viewer_model.Escape_typed()

        @viewer.bind_key("Enter", overwrite=True)
        def Enter_typed(event):
            logger.info("Enter typed")
            self.viewer_model.Enter_typed()

        @viewer.bind_key("r", overwrite=True)
        def r_typed(event):
            logger.info("r typed")
            self.viewer_model.r_typed()

        @viewer.bind_key("s", overwrite=True)
        def s_typed(event):
            logger.info("s typed")
            self.viewer_model.s_typed()

        @viewer.bind_key("d", overwrite=True)
        def d_typed(event):
            logger.info("d typed")
            self.viewer_model.d_typed()

        @viewer.bind_key("t", overwrite=True)
        def t_typed(event):
            logger.info("t typed")
            self.viewer_model.t_typed()

        @log_error
        @viewer.bind_key("Control-Alt-Shift-S", overwrite=True)
        def save_typed(event):
            logger.info("saving validation results...")
            global viewer_model
            save_results(zarr_path.replace(".zarr","_travari.zarr"),
                         label_layer.data,
                         chunks,
                         _df_segments,_df_divisions,
                         self.viewer_model.finalized_segment_ids,
                         self.viewer_model.candidate_segment_ids)
            logger.info("done.")