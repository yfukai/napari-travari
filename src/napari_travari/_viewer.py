import napari
import numpy as np
from dask import array as da
from transitions import Machine
from ._viewer_model import ViewerModel, ViewerState
from ._transitions import transitions
from ._logging import log_error, logger

def load_data(zarr_path):
    return 

# XXX tenative implementation : pluginfy later.
class TravariViewer:
    def __init__(self, image, mask, 
                 zarr_path,
                 data_chunk,
                 new_segment_id = None, 
                 new_label_value = None,
                 finalized_segment_ids = None,
                 candidate_segment_ids = None,
                 ):


        self.viewer = napari.Viewer()
        contrast_limits = np.percentile(np.array(image[0]).ravel(), (50, 98))
        self.viewer.add_image(image, contrast_limits=contrast_limits)
        self.label_layer = self.viewer.add_labels(mask, name="Mask")
        self.sel_label_layer = self.viewer.add_labels(
            da.zeros_like(mask, dtype=np.uint8), name="Selected Mask"
        )
        self.sel_label_layer.contour=3
        self.redraw_label_layer = self.viewer.add_labels(
            np.zeros(mask.shape[-3:], dtype=np.uint8), name="Drawing"
        )
        self.finalized_label_layer = self.viewer.add_labels(
            da.zeros_like(mask, dtype=np.uint8), name="Finalized",
           # color ={1:"red"}, not working
            opacity=1.0,blending="opaque"
        )
        self.finalized_label_layer.contour=3

        self.viewer_model = ViewerModel(
            self,
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

        self.label_layer.mouse_drag_callbacks.append(track_clicked)

        bind_keys=["Escape","Enter","r","s","d","t",]
        for k in bind_keys:
            @log_error
            def typed(_event):
                logger.info(f"{k} typed")
                getattr(self.viewer_model,f"{k}_typed")() # call associated function of the model
            #register the callback to the viewer
            self.viewer.bind_key(k,typed,overwrite=True)

        @log_error
        def save_typed(_event):
            logger.info("saving validation results...")
            self.viewer_model.save_results(self.zarr_path.replace(".zarr","_travari.zarr"), self.data_chunks)
            logger.info("done.")
        self.viewer.bind_key("Control-Alt-Shift-S", save_typed, overwrite=True)