from enum import Enum
from os import path
import numpy as np
import dask.array as da
import zarr
import pandas as pd
import networkx as nx
from ._logging import logger,log_error
from ._consts import *
from ._gui_utils import ask_draw_label, choose_direction_by_mbox, choose_division_by_mbox, get_annotation_of_track_end, ask_ok_or_not

class ViewerState(Enum):
    ALL_LABEL = 1
    LABEL_SELECTED = 2
    LABEL_REDRAW = 3
    LABEL_SWITCH = 4
    DAUGHTER_SWITCH = 5
    DAUGHTER_DRAW = 6
    DAUGHTER_CHOOSE_MODE = 7

class ViewerModel:
    def __init__(self,
                 travari_viewer,
                 df_segments,
                 df_divisions,
                *,
                 new_segment_id,
                 new_label_value,
                 finalized_segment_ids,
                 candidate_segment_ids,
                 ):
        self.selected_label = None
        self.segment_id = None
        self.frame_childs = None
        self.label_childs = None
        self.segment_labels = None
        self.label_edited = None
        self.termination_annotation = ""

        self.target_Ts = list(travari_viewer.target_Ts)
        self.viewer=travari_viewer.viewer
        self.label_layer=travari_viewer.label_layer
        self.redraw_label_layer=travari_viewer.redraw_label_layer
        self.sel_label_layer=travari_viewer.sel_label_layer
        self.finalized_label_layer=travari_viewer.finalized_label_layer
        self.shape=self.label_layer.data.shape
        self.sizeT=self.label_layer.data.shape[0]

        self.df_segments=df_segments
        self.df_divisions=df_divisions
        self.new_segment_id = new_segment_id
        self.new_label_value = new_label_value
        self.finalized_segment_ids = finalized_segment_ids
        self.candidate_segment_ids = candidate_segment_ids
        self.finalized_label_layer.data=self.label_layer.data.map_blocks(
            self.__mask_to_finalized_mask, dtype=np.uint8
        )

        self.viewer_state_active = {
            ViewerState.ALL_LABEL: self.label_layer,
            ViewerState.LABEL_SELECTED: self.sel_label_layer,
            ViewerState.LABEL_REDRAW: self.redraw_label_layer,
            ViewerState.LABEL_SWITCH: self.label_layer,
            ViewerState.DAUGHTER_SWITCH: self.label_layer,
            ViewerState.DAUGHTER_DRAW: self.redraw_label_layer,
            ViewerState.DAUGHTER_CHOOSE_MODE: self.sel_label_layer
        }
        self.layers = [self.label_layer, self.sel_label_layer, 
                       self.redraw_label_layer, self.finalized_label_layer]
        self.viewer_state_visibility = {
            ViewerState.ALL_LABEL: [True, False, False, True],
            ViewerState.LABEL_SELECTED: [True, True, False, False],
            ViewerState.LABEL_REDRAW: [False, False, True, False],
            ViewerState.LABEL_SWITCH: [True, False, False, True],
            ViewerState.DAUGHTER_SWITCH: [True, False, False, True],
            ViewerState.DAUGHTER_DRAW: [False, False, True, False],
            ViewerState.DAUGHTER_CHOOSE_MODE: [False, True, False, False],
        }

    @log_error
    def update_layer_status(self,*_):
        visibles=self.viewer_state_visibility[self.state]
        assert len(visibles) == len(self.layers)
        for i in range(len(self.layers)):
            try:
                self.layers[i].visible = visibles[i]
            except ValueError:
                pass

        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(self.viewer_state_active[self.state])

    @log_error
    def refresh_redraw_label_layer(self):
        self.redraw_label_layer.data = np.zeros_like(self.redraw_label_layer.data)
        self.redraw_label_layer.mode = "paint"

    @log_error
    def __mask_to_selected_label_image(self, block, block_info=None):
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
                    indices = [slice(loc[0], loc[1]) for loc in location]
                    sub_label = label[tuple(indices)[2:]]
                    sel_mask[0, 0][sub_label] = j + 2
        return sel_mask

    @log_error
    def __mask_to_finalized_mask(self, block, block_info=None):
        #        print("block_info",block_info[0]['array-location'])
        if block_info is None or len(block_info) == 0:
            return None
        location = block_info[0]["array-location"]
        frame = location[0][0]
        try:
            segments_at_frame = self.df_segments.loc[frame]
        except KeyError:
            return np.zeros_like(block,dtype=np.uint8)

        finalized_labels_at_frame = segments_at_frame[
            segments_at_frame["segment_id"].isin(self.finalized_segment_ids)
        ].index.get_level_values("label")

        candidate_labels_at_frame = segments_at_frame[
            segments_at_frame["segment_id"].isin(self.candidate_segment_ids)
        ].index.get_level_values("label")

        mask_finalized = (np.isin(block,np.array(finalized_labels_at_frame))).astype(np.uint8)
        mask_candidate = (np.isin(block,np.array(candidate_labels_at_frame))).astype(np.uint8)
        return mask_finalized + 2*mask_candidate


    @log_error
    def select_track(self, frame, val, segment_id):
        self.segment_id = segment_id
        segment_labels = np.ones(self.sizeT, dtype=np.uint32) * NOSEL_VALUE
        df = self.df_segments[self.df_segments["segment_id"] == segment_id]
        frames = df.index.get_level_values("frame").values
        labels = df.index.get_level_values("label").values
        segment_labels[frames] = labels

        self.label_edited = np.zeros(len(segment_labels), dtype=bool)
        self.segment_labels = segment_labels
        self.original_segment_labels = segment_labels.copy()
        self.label_layer.termination_annotation = ""
        # used to rewrite track on exit

        row = self.df_divisions[self.df_divisions["segment_id_parent"] == segment_id]
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
        self.sel_label_layer.data = self.label_layer.data.map_blocks(
            self.__mask_to_selected_label_image, dtype=np.uint8
        )

    @log_error
    def label_redraw_enter_valid(self):
        iT = self.viewer.dims.current_step[0]
        #return True if:
        # - this timeframe is in target_T 
        # - segment_labels is not NOSEL_VALUE in either of this, previous, next target_T
        if not iT in self.target_Ts:
            logger.info("this frame is not in target_Ts")
            return False
        previous_iT=self.target_Ts[max(0,self.target_Ts.index(iT)-1)]
        next_iT=self.target_Ts[min(len(self.target_Ts)-1,self.target_Ts.index(iT)+1)]
        if (    self.segment_labels[         iT] == NOSEL_VALUE 
            and self.segment_labels[previous_iT] == NOSEL_VALUE 
            and self.segment_labels[    next_iT] == NOSEL_VALUE 
#            not np.any(self.sel_label_layer.data[iT] == 1)
#            and not np.any(self.sel_label_layer.data[min(iT + 1, self.sizeT)] == 1)
#            and not np.any(self.sel_label_layer.data[max(iT - 1, 0)] == 1)
        ):
            logger.info("track does not exist in connected timeframe")
            return False
        else:
            logger.info("redraw valid")
            return True

    @log_error
    def check_drawn_label(self):
        return np.any(self.redraw_label_layer.data == 1)

    @log_error
    def label_redraw_finish(self):
        logger.info("label redraw finish")
        iT = self.viewer.dims.current_step[0]
        logger.info("label redraw finish")
        self.sel_label_layer.data[iT] = 0
        self.sel_label_layer.data[iT] = self.redraw_label_layer.data == 1
        self.label_edited[iT] = True
        if self.segment_labels[iT] == NOSEL_VALUE:
            self.segment_labels[iT] = NEW_LABEL_VALUE
        else:
            if ask_draw_label(self.viewer) == "new":
                self.segment_labels[iT] = NEW_LABEL_VALUE
            

    @log_error
    def switch_track_enter_valid(self):
        iT = self.viewer.dims.current_step[0]
        if not iT in self.target_Ts:
            logger.info("this frame is not in target_Ts")
            return False
        previous_iT=self.target_Ts[max(0,self.target_Ts.index(iT)-1)]
        next_iT=self.target_Ts[min(len(self.target_Ts)-1,self.target_Ts.index(iT)+1)]
        if (    self.segment_labels[         iT] == NOSEL_VALUE 
            and self.segment_labels[previous_iT] == NOSEL_VALUE 
            and self.segment_labels[    next_iT] == NOSEL_VALUE 
#            not np.any(self.sel_label_layer.data[iT] == 1)
#            and not np.any(self.sel_label_layer.data[min(iT + 1, self.sizeT)] == 1)
#            and not np.any(self.sel_label_layer.data[max(iT - 1, 0)] == 1)
        ):
            logger.info("track does not exist in connected timeframe")
            return False
        else:
            logger.info("switch valid")
            return True

    @log_error
    def switch_track(self, frame, val, segment_id):
        direction = choose_direction_by_mbox(self.viewer)

        if not direction:
            return
        elif direction == "forward":
            print("forward ... ")
            df = self.df_segments[
                (self.df_segments["segment_id"] == segment_id)
                & (self.df_segments.index.get_level_values("frame") >= frame)
            ]
            frames = df.index.get_level_values("frame").values
            labels = df.index.get_level_values("label").values

            self.segment_labels[frame:] = NOSEL_VALUE
            self.segment_labels[frames] = labels
            self.label_edited[frame:] = False
            #FIXME revert layer to original
            row = self.df_divisions[self.df_divisions["segment_id_parent"] == segment_id]

            if len(row) == 1:
                self.frame_childs = row.iloc[0][["frame_child1", "frame_child2"]]
                self.label_childs = row.iloc[0][["label_child1", "label_child2"]]
            elif len(row) == 0:
                self.frame_childs = []
                self.label_childs = []
            self.termination_annotation = ""

        elif direction == "backward":
            df = self.df_segments[
                (self.df_segments["segment_id"] == segment_id)
                & (self.df_segments.index.get_level_values("frame") <= frame)
            ]
            frames = df.index.get_level_values("frame").values
            labels = df.index.get_level_values("label").values
            self.segment_labels[:frame] = NOSEL_VALUE
            self.segment_labels[frames] = labels
            self.label_edited[:frame] = False

    @log_error
    def daughter_choose_mode_enter_valid(self):
        logger.info("enter daughter choose")
        iT = self.viewer.dims.current_step[0]
        if not iT in self.target_Ts:
            logger.info("this frame is not in target_Ts")
            return False
        previous_iT=self.target_Ts[max(0,self.target_Ts.index(iT)-1)]
        if (    self.segment_labels[         iT] == NOSEL_VALUE 
            and self.segment_labels[previous_iT] == NOSEL_VALUE 
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
            method = choose_division_by_mbox(self.viewer)
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
        self.label_child_candidates.append(self.redraw_label_layer.data == 1)

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
        iT = self.viewer.dims.current_step[0]
        if not np.any(self.sel_label_layer.data[iT] == 1):
            logger.info("track does not exist in connected timeframe")
            return False
        else:
            logger.info("marking termination valid")
            return True

    @log_error
    def mark_termination(self):
        iT = self.viewer.dims.current_step[0]
        termination_annotation, res = get_annotation_of_track_end(self.viewer)
        if res:
            logger.info("marking termination: {termination_annotation}")
            self.termination_annotation = termination_annotation
            if iT < self.segment_labels.shape[0]-1:
                self.segment_labels[iT+1:] = NOSEL_VALUE
        else:
            logger.info("marking termination cancelled")

    @log_error
    def finalize_track(self):
        segment_id = self.segment_id
        segment_labels = self.segment_labels
        label_edited = self.label_edited

        frame_childs = self.frame_childs.copy()
        label_childs = self.label_childs.copy()

        termination_annotation = self.termination_annotation

        segment_graph=nx.Graph()
        frame_labels=list(enumerate(segment_labels))+list(zip(frame_childs,label_childs))
        relevant_segment_ids = np.unique([
            self.df_segments.loc[(frame,label),"segment_id"] 
            for frame,label in frame_labels 
            if np.isscalar(label) and label != NOSEL_VALUE and label != NEW_LABEL_VALUE ])
        
        last_frames={}
        for relevant_segment_id in relevant_segment_ids:
            df=self.df_segments[self.df_segments["segment_id"]==relevant_segment_id] 
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
            self.df_segments.loc[(frame,label),"segment_id"] = segment_id
        
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
            original_segment_id = self.df_segments.loc[frame_labels,"segment_id"]
            assert np.all(original_segment_id.iloc[0]==original_segment_id)
            original_segment_id=original_segment_id.iloc[0]
            last_frame = last_frames[original_segment_id]
            frames, _ = zip(*frame_labels)
            
            self.df_segments.loc[frame_labels,"segment_id"] = self.new_segment_id
            if np.any(frames==last_frame):
                ind = self.df_divisions["segment_id_parent"] == original_segment_id
                if np.any(ind):
                    assert np.sum(ind) == 1
                    self.df_divisions.loc[ind,"segment_id_parent"] = self.new_segment_id
            self.new_segment_id += 1

        def __draw_label(mask_image,frame,label):
            #XXX tenative imprementation, faster if directly edit the zarr?
            __dask_compute = lambda arr : arr.compute() if isinstance(arr,da.Array) else arr
            inds = [__dask_compute(i) for i in np.where(mask_image)]
            bboxes = [(np.min(ind),np.max(ind)+1) for ind in inds]
            subimg = np.array(self.label_layer.data[
                frame,0,0,slice(*bboxes[0]),slice(*bboxes[1])])
            subimg[tuple((ind-bbox[0]) for ind,bbox in zip(inds,bboxes))] = label
            self.label_layer.data[frame,0,0,slice(*bboxes[0]),slice(*bboxes[1])]=subimg
            return bboxes
            

        for redrawn_frame in np.where(self.label_edited)[0]:
            label = self.segment_labels[redrawn_frame]
            if not label in [NOSEL_VALUE,NEW_LABEL_VALUE]:
                __draw_label(self.label_layer.data[redrawn_frame,0,0]==label,redrawn_frame,0)
            else:
                label=self.new_label_value
                self.df_segments=self.df_segments.append(pd.Series(
                    {"segment_id":segment_id},name=(redrawn_frame,label)
                ))
                self.new_label_value+=1
            
            bboxes = __draw_label(self.sel_label_layer.data[redrawn_frame,0,0]==1,
                                  redrawn_frame,label)
            # set bounding box
            self.df_segments.loc[(redrawn_frame,label),"bbox_y0"]=bboxes[0][0]
            self.df_segments.loc[(redrawn_frame,label),"bbox_y1"]=bboxes[0][1]
            self.df_segments.loc[(redrawn_frame,label),"bbox_x0"]=bboxes[1][0]
            self.df_segments.loc[(redrawn_frame,label),"bbox_x1"]=bboxes[1][1]

        ind = self.df_divisions["segment_id_parent"] == segment_id
        if np.any(ind):
            assert np.sum(ind) == 1
            self.df_divisions=self.df_divisions[~ind]
            self.new_segment_id += 1

        if len(frame_childs) > 0 and len(label_childs) > 0:
            assert len(frame_childs) == 2 and len(label_childs) == 2 
            division_row={
                "segment_id_parent" : segment_id
            }
            for j, (frame_child, label_child) in enumerate(zip(frame_childs,label_childs)):
                division_row[f"frame_child{j+1}"]=frame_child
                if np.isscalar(label_child):
                    #means the daughter was selected
                    division_row[f"label_child{j+1}"]=label_child
                    segment_id_child=self.df_segments.loc[(frame_child,label_child),
                                                      "segment_id"]
                else:
                    bboxes = __draw_label(label_child[0],
                                          frame_child,
                                          self.new_label_value)
                    division_row[f"label_child{j+1}"]=self.new_label_value
                    self.df_segments=self.df_segments.append(pd.Series(
                        {
                            "segment_id":self.new_segment_id,
                            "bbox_y0":bboxes[0][0],
                            "bbox_y1":bboxes[0][1],
                            "bbox_x0":bboxes[1][0],
                            "bbox_x1":bboxes[1][1],
                        },
                        name=(frame_child,self.new_label_value)
                    ))
                    segment_id_child=self.new_segment_id
                    self.new_segment_id+=1
                    self.new_label_value+=1
                if not segment_id_child in self.finalized_segment_ids:
                    logger.info(f"candidate adding ... {segment_id_child}")
                    self.candidate_segment_ids.add(segment_id_child)
            self.df_divisions=self.df_divisions.append(division_row,ignore_index=True)

        self.finalized_segment_ids.add(segment_id)
        self.candidate_segment_ids.discard(segment_id)

        self.finalized_label_layer.data=self.label_layer.data.map_blocks(
            self.__mask_to_finalized_mask, dtype=np.uint8
        )

    @log_error
    def save_results(self,zarr_path,chunks):
        logger.info("saving validation results...")
        if path.exists(zarr_path):
            if_overwrite = ask_ok_or_not(self.viewer,"Validation file already exists. Overwrite?")
            if not if_overwrite:
                return
        logger.info("saving...")
        # to avoid IO from/to the same array, save to a temp array and then rename
        self.label_layer.data.rechunk(chunks).to_zarr(zarr_path,"mask_tmp",overwrite=True)
        logger.info("saving...")
        zarr_file=zarr.open(zarr_path,"a")
        del zarr_file["mask"]
        zarr_file.store.rename("mask_tmp","mask")

        mask_ds=zarr_file["mask"]
        logger.info("saving...")
        #XXX dirty implementation for now as those dataframes are not heavy
        mask_ds.attrs["df_segments"]=self.df_segments.to_csv()
        logger.info("saving...")
        mask_ds.attrs["df_divisions"]=self.df_divisions.to_csv()
        logger.info("saving...")
        mask_ds.attrs["finalized_segment_ids"]=list(map(int,self.finalized_segment_ids))
        mask_ds.attrs["candidate_segment_ids"]=list(map(int,self.candidate_segment_ids))
        mask_ds.attrs["target_Ts"]=list(map(int,self.target_Ts))

        logger.info("saving...")
        self.label_layer.data = da.from_zarr(mask_ds).persist()
        logger.info("saving validation results...")
    
