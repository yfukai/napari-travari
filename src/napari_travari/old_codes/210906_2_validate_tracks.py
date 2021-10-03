#%%
#base_dir = "D:\Fukai-ImageAnalysis\old\LSM800_2021-03-04-timelapse_old"
#%%
#base_dir = "/Volumes/Extreme SSD/LSM800_2021-03-04-timelapse_old"
#%%
base_dir = "/home/fukai/microscope_analysis/old/LSM800_2021-03-04-timelapse_old"
#%%
from ctypes import Array
import logging
from enum import Enum
from time import sleep
from os import path
import traceback
from functools import wraps
from itertools import chain

import pandas as pd
import numpy as np
import dask
import dask.array as da
import zarr
import napari
import networkx as nx

from tqdm import tqdm
from transitions import Machine

from qtpy.QtWidgets import QMessageBox, QInputDialog

print(dask.__version__)
print(napari.__version__)

log_path=path.join(path.expanduser("~"),".tracking_log/log.txt")
logging.basicConfig(filename=log_path,
                    level=logging.INFO)
logger=logging.getLogger(__name__)

NOSEL_VALUE = np.iinfo(np.uint32).max
NEW_LABEL_VALUE = NOSEL_VALUE - 1
zarr_path = path.join(base_dir, "image_total_aligned_small2.zarr")

#class RedoableZarr:
#    def __init__(self,zarr_file,name,
#                 suffix="_history",
#                 index_suffix="_history_index",
#                 init_history_size=(10,5000)):
#        self.zarr_file = zarr_file
#        self.name = name
#        self.history_data_name=self.name+suffix
#        self.index_data_name=self.name+index_suffix
#
#        self.data = self.zarr_file[self.name]
#        self.history_data = self.zarr_file.require_dataset(
#            self.history_data_name,
#            shape=init_history_size,
#            dtype=np.uint64,
#            )
#        self.index_data = self.zarr_file.require_dataset(
#            self.index_data_name,
#            shape=init_history_size,
#            dtype=np.uint64,
#            )
#        self.history_count=0
#
#     def __setitem__(self,ind,value):
#        self.zarr_file[ind]


#%%
logger.info("program started")

#%%
zarr_file = zarr.open(zarr_path, "a")

image = da.from_zarr(zarr_file["image"])
mask = da.from_zarr(zarr_file["mask"])[:, np.newaxis, ...]
sizeT = mask.shape[0]
mask[mask == -1] = 0 # mask.max().compute() + 1

segment_columns = ["segment_id","bbox_y0","bbox_y1","bbox_x0","bbox_x1"]
df_segments2 = pd.read_csv(
    path.join(base_dir, "df_segments2_updated.csv"), 
    index_col=["frame", "label"],
#    dtype = pd.Int64Dtype()
)[segment_columns]
df_segments2=df_segments2.astype(
    dict(zip(segment_columns,[pd.Int64Dtype()]+[pd.Int32Dtype()]*4))
)

division_columns=["segment_id_parent","frame_child1","label_child1","frame_child2","label_child2"]
df_divisions2 = pd.read_csv(
    path.join(base_dir, "df_divisions2.csv"), 
    index_col=0,
#    dtype = pd.Int64Dtype()
)[division_columns]
df_divisions2=df_divisions2.astype(pd.Int64Dtype())


#segment_labels = df_segments2.xs(
#    0, level="frame", drop_level=False
#).index.get_level_values("label")
#labels = [0] + sorted(list(set(segment_labels.values)))

new_label_value = df_segments2.index.get_level_values("label").max() + 1
assert not np.any(mask ==new_label_value)
new_segment_id = df_segments2["segment_id"].max() + 1

#%%
viewer = napari.Viewer()
contrast_limits = np.percentile(np.array(image[0]).ravel(), (50, 98))
viewer.add_image(image, contrast_limits=contrast_limits)
label_layer = viewer.add_labels(mask, name="Mask")
sel_label_layer = viewer.add_labels(
    da.zeros_like(mask, dtype=np.uint8), name="Selected Mask"
)
redraw_label_layer = viewer.add_labels(
    np.zeros(mask.shape[-3:], dtype=np.uint8), name="Drawing"
)
finalized_label_layer = viewer.add_labels(
    da.zeros_like(mask, dtype=np.uint8), name="Finalized"
)
finalized_tracks = [[]] * mask.shape[0]

layers = [label_layer, sel_label_layer, redraw_label_layer, finalized_label_layer]


def set_visible_layers(visibles):
    assert len(visibles) == len(layers)
    for i in range(len(layers)):
        try:
            layers[i].visible = visibles[i]
        except ValueError:
            pass

#%%
#%%
_df_segments=df_segments2.copy()
_df_divisions=df_divisions2.copy()
_new_segment_id=new_segment_id

#%%


def log_error(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args,**kwargs)
        except BaseException as error:
            logger.error(traceback.format_exc())
    return wrapped
        

class ViewerModel:
    def __init__(self,*,new_segment_id,new_label_value):
        self.selected_label = None
        self.segment_id = None
        self.frame_childs = None
        self.label_childs = None
        self.segment_labels = None
        self.label_edited = None
        self.termination_annotation = ""
        self.new_segment_id = new_segment_id
        self.new_label_value = new_label_value

    @log_error
    def update_layer_status(self,*_):
        set_visible_layers(viewer_state_visibility[self.state])
        viewer.layers.selection.clear()
        viewer.layers.selection.add(viewer_state_active[self.state])

    @log_error
    def refresh_redraw_label_layer(self):
        redraw_label_layer.data = np.zeros_like(redraw_label_layer.data)
        redraw_label_layer.mode = "paint"

    @log_error
    def to_selected_label_image(self, block, block_info=None):
        #        print("block_info",block_info[0]['array-location'])
        assert not self.segment_labels is None
        assert not self.frame_childs is None
        assert not self.label_childs is None
        if block_info is None or len(block_info) == 0:
            return None
        location = block_info[0]["array-location"]
        iT = location[0][0]
        sel_mask = (block == self.segment_labels[iT]).astype(np.uint8)
        # reading from df_segments2
        for j, (frame, label) in enumerate(zip(self.frame_childs, self.label_childs)):
            if iT == frame:
                if np.isscalar(label):
                    sel_mask[block == label] = j + 2
                else:
                    #                    print("sel_mask",location,sel_mask.shape,label.shape)
                    indices = [slice(loc[0], loc[1]) for loc in location]
                    sub_label = label[tuple(indices)[2:]]
                    #                    print(sub_label.shape)
                    sel_mask[0, 0][sub_label] = j + 2
        return sel_mask

    @log_error
    def select_track(self, frame, val, segment_id):
        self.segment_id = segment_id
        segment_labels = np.ones(image.shape[0], dtype=np.uint32) * NOSEL_VALUE
        df = _df_segments[_df_segments["segment_id"] == segment_id]
        frames = df.index.get_level_values("frame").values
        labels = df.index.get_level_values("label").values
        segment_labels[frames] = labels

        self.label_edited = np.zeros(len(segment_labels), dtype=bool)
        self.segment_labels = segment_labels
        self.original_segment_labels = segment_labels.copy()
        label_layer.termination_annotation = ""
        # used to rewrite track on exit

        row = _df_divisions[_df_divisions["segment_id_parent"] == segment_id]
        print(row)
        if len(row) == 1:
            self.frame_childs = list(row.iloc[0][["frame_child1", "frame_child2"]])
            self.label_childs = list(row.iloc[0][["label_child1", "label_child2"]])
        elif len(row) == 0:
            self.frame_childs = []
            self.label_childs = []
        else:
            return
        print(self.frame_childs, self.label_childs)
        sel_label_layer.data = label_layer.data.map_blocks(
            self.to_selected_label_image, dtype=np.uint8
        )

    @log_error
    def label_redraw_enter_valid(self):
        iT = viewer.dims.current_step[0]
        if (
            not np.any(sel_label_layer.data[iT] == 1)
            and not np.any(sel_label_layer.data[min(iT + 1, sizeT)] == 1)
            and not np.any(sel_label_layer.data[max(iT - 1, 0)] == 1)
        ):
            logger.info("track does not exist in connected timeframe")
            return False
        else:
            logger.info("redraw valid")
            return True

    @log_error
    def check_drawn_label(self):
        return np.any(redraw_label_layer.data == 1)

    @log_error
    def label_redraw_finish(self):
        logger.info("label redraw finish")
        iT = viewer.dims.current_step[0]
        logger.info("label redraw finish")
        sel_label_layer.data[iT] = 0
        sel_label_layer.data[iT] = redraw_label_layer.data == 1
        self.label_edited[iT] = True
        if self.segment_labels[iT] == NOSEL_VALUE:
            self.segment_labels[iT] = NEW_LABEL_VALUE

    @log_error
    def switch_track_enter_valid(self):
        iT = viewer.dims.current_step[0]
        if (
            not np.any(sel_label_layer.data[iT] == 1)
            and not np.any(sel_label_layer.data[min(iT + 1, sizeT)] == 1)
            and not np.any(sel_label_layer.data[max(iT - 1, 0)] == 1)
        ):
            logger.info("track does not exist in connected timeframe")
            return False
        else:
            logger.info("switch valid")
            return True

    @log_error
    def switch_track(self, frame, val, segment_id):
        direction = choose_direction_by_mbox(viewer)

        if not direction:
            return
        elif direction == "forward":
            print("forward ... ")
            df = _df_segments[
                (_df_segments["segment_id"] == segment_id)
                & (_df_segments.index.get_level_values("frame") >= frame)
            ]
            frames = df.index.get_level_values("frame").values
            labels = df.index.get_level_values("label").values

            self.segment_labels[frame:] = NOSEL_VALUE
            self.segment_labels[frames] = labels
            self.label_edited[frame:] = False
            row = _df_divisions[_df_divisions["segment_id_parent"] == segment_id]

            if len(row) == 1:
                self.frame_childs = row.iloc[0][["frame_child1", "frame_child2"]]
                self.label_childs = row.iloc[0][["label_child1", "label_child2"]]
            elif len(row) == 0:
                self.frame_childs = []
                self.label_childs = []
            self.termination_annotation = ""

        elif direction == "backward":
            df = _df_segments[
                (_df_segments["segment_id"] == segment_id)
                & (_df_segments.index.get_level_values("frame") <= frame)
            ]
            frames = df.index.get_level_values("frame").values
            labels = df.index.get_level_values("label").values
            self.segment_labels[:frame] = NOSEL_VALUE
            self.segment_labels[frames] = labels
            self.label_edited[:frame] = False

    @log_error
    def daughter_choose_mode_enter_valid(self):
        logger.info("enter daughter choose")
        iT = viewer.dims.current_step[0]
        if not np.any(sel_label_layer.data[iT] == 1) and not np.any(
            sel_label_layer.data[max(iT - 1, 0)] == 1
        ):
            logger.info("track does not exist in connected timeframe")
            return False
        logger.info("mark division...")
        self.frame_child_candidate = iT
        self.label_child_candidates = []
        return True

    @log_error
    def on_enter_DAUGHTER_CHOOSE_MODE(self,*_):
        logger.info("candidates count: %i",len(self.label_child_candidates))
        if len(self.label_child_candidates) == 2:
            self.finalize_daughter()
            self.to_LABEL_SELECTED()
        else:
            method = choose_division_by_mbox(viewer)
            logger.info("%s selected", method)
            if method == "select":
                self.to_DAUGHTER_SWITCH()
            elif method == "draw":
                self.refresh_redraw_label_layer()
                self.to_DAUGHTER_DRAW()
            else:
                self.to_LABEL_SELECTED()

    @log_error
    def daughter_select(self, frame, val, segment_id):
        if frame == self.frame_child_candidate:
            self.label_child_candidates.append(int(val))
        else: 
            logger.info("frame not correct")

    @log_error
    def daughter_draw_finish(self):
        self.label_child_candidates.append(redraw_label_layer.data == 1)

    @log_error
    def finalize_daughter(self):
        assert len(self.label_child_candidates) == 2
        self.frame_childs = []
        self.label_childs = []
        for j, candidate in enumerate(self.label_child_candidates):
            self.label_childs.append(candidate)
            self.frame_childs.append(self.frame_child_candidate)
        self.segment_labels[self.frame_child_candidate :] = NOSEL_VALUE

    @log_error
    def mark_termination_enter_valid(self):
        iT = viewer.dims.current_step[0]
        if not np.any(sel_label_layer.data[iT] == 1):
            logger.info("track does not exist in connected timeframe")
            return False
        else:
            logger.info("marking termination valid")
            return True

    @log_error
    def mark_termination(self):
        iT = viewer.dims.current_step[0]
        termination_annotation, res = get_annotation_of_track_end(viewer)
        if res:
            logger.info("marking termination: {termination_annotation}")
            self.termination_annotation = termination_annotation
            if iT < self.segment_labels.shape[0]-1:
                self.segment_labels[iT+1:] = NOSEL_VALUE
        else:
            logger.info("marking termination cancelled")

    @log_error
    def finalize_track(self):
        global _df_segments, _df_divisions
        segment_id = self.segment_id
        segment_labels = self.segment_labels
        label_edited = self.label_edited

        frame_childs = self.frame_childs.copy()
        label_childs = self.label_childs.copy()

        termination_annotation = self.termination_annotation

        segment_graph=nx.Graph()
        frame_labels=list(enumerate(segment_labels))+list(zip(frame_childs,label_childs))
        relevant_segment_ids = np.unique([
            _df_segments.loc[(frame,label),"segment_id"] 
            for frame,label in frame_labels 
            if np.isscalar(label) and label != NOSEL_VALUE and label != NEW_LABEL_VALUE ])
        
        last_frames={}
        for relevant_segment_id in relevant_segment_ids:
            df=_df_segments[_df_segments["segment_id"]==relevant_segment_id] 
            if len(df) ==0:
                continue
            df=df.sort_index(level='frame')
            last_frames[relevant_segment_id]=df.index.get_level_values("frame")[-1]
            if len(df) == 1:
                frame,label=df.index[0]
                segment_graph.add_node((frame,label))
            else:
                for ((frame1,label1),_),((frame2,label2),_) in \
                    zip(df.iloc[:-1].iterrows(),df.iloc[1:].iterrows()):
                    segment_graph.add_edge((frame1,label1),(frame2,label2))
        
        for frame,label in enumerate(segment_labels):
            if label in (NOSEL_VALUE, NEW_LABEL_VALUE): continue
            segment_graph.remove_node((frame,label)) 
            _df_segments.loc[(frame,label),"segment_id"] = segment_id
        
        for frame,label in zip(frame_childs,label_childs):
            if not np.isscalar(label):
                continue
            neighbors = segment_graph.neighbors((frame,label))
            ancestors = [n for n in neighbors if n[0]<frame]
            if len(ancestors)==0:
                continue
            else:
                assert len(ancestors)==1
                ancestor = ancestors[0]
                segment_graph.remove_edge((frame,label),ancestor)
        
        # relavel divided tracks
        for subsegment in nx.connected_components(segment_graph):
            frame_labels=sorted(subsegment,key=lambda x :x[0])
            original_segment_id = _df_segments.loc[frame_labels,"segment_id"]
            assert np.all(original_segment_id.iloc[0]==original_segment_id)
            original_segment_id=original_segment_id.iloc[0]
            last_frame = last_frames[original_segment_id]
            frames, _ = zip(*frame_labels)
            
            _df_segments.loc[frame_labels,"segment_id"] = self.new_segment_id
            if np.any(frames==last_frame):
                ind = _df_divisions["segment_id_parent"] == original_segment_id
                if np.any(ind):
                    assert np.sum(ind) == 1
                    _df_divisions.loc[ind,"segment_id_parent"] = self.new_segment_id
            self.new_segment_id += 1

        def __draw_label(mask_image,frame,label):
            #XXX tenative imprementation
            __dask_compute = lambda arr : arr.compute() if isinstance(arr,da.Array) else arr
            inds = [__dask_compute(i) for i in np.where(mask_image)]
            bboxes = [(np.min(ind),np.max(ind)+1) for ind in inds]
            subimg = np.array(label_layer.data[
                frame,0,0,slice(*bboxes[0]),slice(*bboxes[1])])
            subimg[tuple((ind-bbox[0]) for ind,bbox in zip(inds,bboxes))] = label
            label_layer.data[frame,0,0,slice(*bboxes[0]),slice(*bboxes[1])]=subimg
            return bboxes
            

        for redrawn_frame in np.where(self.label_edited)[0]:
            label = self.segment_labels[redrawn_frame]
            if not label in [NOSEL_VALUE,NEW_LABEL_VALUE]:
                __draw_label(label_layer.data[redrawn_frame,0,0]==label,redrawn_frame,0)
            else:
                label=self.new_label_value
                _df_segments=_df_segments.append(pd.Series(
                    {"segment_id":segment_id},name=(redrawn_frame,label)
                ))
                self.new_label_value+=1
            
            bboxes = __draw_label(sel_label_layer.data[redrawn_frame,0,0]==1,
                                  redrawn_frame,label)
            # set bounding box
            _df_segments.loc[(redrawn_frame,label),
                ["bbox_y0","bbox_y1","bbox_x0","bbox_x1"]]=np.concatenate(bboxes)

        ind = _df_divisions["segment_id_parent"] == segment_id
        if np.any(ind):
            assert np.sum(ind) == 1
            _df_divisions=_df_divisions[~ind]
            self.new_segment_id += 1

        division_row={
            "segment_id_parent" : segment_id
        }
        for j, (frame_child, label_child) in enumerate(zip(frame_childs,label_childs)):
            division_row[f"frame_child{j+1}"]=frame_child
            if np.isscalar(label_child):
                division_row[f"label_child{j+1}"]=label_child
                #means the daughter was selected
            else:
                bboxes = __draw_label(label_child[0],
                                      frame_child,
                                      self.new_label_value)
                division_row[f"label_child{j+1}"]=self.new_label_value
                _df_segments=_df_segments.append(pd.Series(
                    {"segment_id":self.new_segment_id},
                    name=(frame_child,self.new_label_value)
                ))
                self.new_segment_id+=1
                self.new_label_value+=1
        _df_divisions=_df_divisions.append(division_row,ignore_index=True)



viewer_model = ViewerModel(
    new_segment_id=_new_segment_id,
    new_label_value=new_label_value
    )
machine = Machine(
    model=viewer_model,
    states=ViewerState,
    transitions=transitions,
    after_state_change="update_layer_status",
    initial=ViewerState.ALL_LABEL,
)

label_layer.mouse_drag_callbacks.clear()
sel_label_layer.keymap.clear()


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

        row = _df_segments.xs((cords[0], val), level=("frame", "label"))
        if len(row) != 1:
            return
        logger.info(row)
        segment_id = row["segment_id"].iloc[0]

        viewer_model.track_clicked(frame, val ,segment_id)

@viewer.bind_key("Escape", overwrite=True)
def Escape_typed(event):
    global viewer_model
    viewer_model.Escape_typed()

@viewer.bind_key("Enter", overwrite=True)
def Enter_typed(event):
    logger.info("Enter typed")
    global viewer_model
    viewer_model.Enter_typed()

@viewer.bind_key("r", overwrite=True)
def r_typed(event):
    global viewer_model
    viewer_model.r_typed()

@viewer.bind_key("s", overwrite=True)
def s_typed(event):
    global viewer_model
    viewer_model.s_typed()

@viewer.bind_key("d", overwrite=True)
def d_typed(event):
    global viewer_model
    viewer_model.d_typed()

@viewer.bind_key("t", overwrite=True)
def t_typed(event):
    global viewer_model
    viewer_model.t_typed()

viewer_model.update_layer_status()

napari.run()
# #%%
# redrawn_frame=930
# inds = [i.compute() for i in np.where(sel_label_layer.data[redrawn_frame,0,0]==1)]
# #XXX tenative imprementation
# bboxes = [(np.min(ind),np.max(ind)+1) for ind in inds]
# subimg = np.array(label_layer.data[
#     redrawn_frame,0,0,slice(*bboxes[0]),slice(*bboxes[1])])
# subimg[tuple((ind-bbox[0]) for ind,bbox in zip(inds,bboxes))] = 9000
# label_layer.data[redrawn_frame,0,0,slice(*bboxes[0]),slice(*bboxes[1])]=subimg
# 
#segment_id = viewer_model.segment_id
#segment_labels = viewer_model.segment_labels
#label_edited = viewer_model.label_edited
#
#frame_childs = viewer_model.frame_childs.copy()
#label_childs = viewer_model.label_childs.copy()
#
#termination_annotation = viewer_model.termination_annotation
#
#  
## %%
# %%

# %%
_df_segments[_df_segments["segment_id"]==1806]
# %%
