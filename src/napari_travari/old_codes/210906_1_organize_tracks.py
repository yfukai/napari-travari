#%%
import napari
import zarr
import pandas as pd
import numpy as np
import dask.array as da
from tqdm import tqdm
from os import path
import dask
from skimage.measure import regionprops
print(dask.__version__)
print(napari.__version__)
from qtpy.QtWidgets import QMessageBox

#%%
df_segments=pd.read_csv(path.join(base_dir,"tracks_segment_parsed.csv"),index_col=0).reset_index(drop=True)
df_segments=df_segments[["point_id","segment_id","@frame"]].rename(columns={"@frame":"frame"})
display(df_segments[df_segments["segment_id"]==0].tail())
display(df_segments[df_segments["segment_id"]==1].head())

grps=df_segments.groupby("segment_id")
parents_data=[]
before_children_data=[]
children_data=[]
for segment_id, grp in tqdm(grps,total=len(grps)):
    head=grp[grp["frame"]==grp["frame"].min()]
    assert len(head)==1
    head=head.iloc[0]
    df_same_id = df_segments[(df_segments["point_id"]==head["point_id"])&
                             (df_segments["frame"]==head["frame"])]
    if len(df_same_id)>1:
        parent=[]
        children=[]
        for i,row in df_same_id.iterrows():
            df=df_segments[df_segments["segment_id"]==row["segment_id"]]
            if df[df["frame"]==df["frame"].min()].index[0]==i:
                children.append(i)
            else:
                parent.append(i)
        assert len(children)==2 and len(parent)==1
        parents_data.append(parent[0])
        children_data.append(children)
        #df_segments=df_segments.drop(index=children)
children_data=np.array(children_data)
df_divisions = pd.DataFrame({
   "parent_id": parents_data,
   "before_child1_id" : children_data[:,0],
   "before_child2_id" : children_data[:,1],
},dtype=np.uint32)

def get_next_val(ind):
    row=df_segments.loc[ind]
    row2=df_segments[
        (df_segments["segment_id"]==row["segment_id"])&\
        (df_segments["frame"]==row["frame"]+1)
    ]
    assert len(row2)==1
    return row2.index[0]


df_divisions["child1_id"]=df_divisions["before_child1_id"].apply(get_next_val)
df_divisions["child2_id"]=df_divisions["before_child2_id"].apply(get_next_val)
df_divisions=df_divisions.drop_duplicates()

del_ids=list(set(df_divisions[["before_child1_id","before_child2_id"]].values.ravel()))
df_segments2=df_segments.drop(index=del_ids)
df_segments2["label"]=df_segments2["segment_id"]+1
df_divisions2=df_divisions.drop(columns=["before_child1_id","before_child2_id"])

for k in ["parent","child1","child2"]:
    print(k)
    df=df_segments2.copy()
    df.columns=df.columns+f"_{k}"
    df_divisions2=pd.merge(df_divisions2,df,
                           left_on=f"{k}_id",
                           right_index=True)
#assert len(df_segments2.groupby(["frame","label"]))==len(df_segments2)
df_segments2=df_segments2.set_index(["frame","label"],append=True)

df_segments2=pd.read_csv(path.join(base_dir,"df_segments2.csv"),index_col=[0,"frame","label"])
df_divisions2=pd.read_csv(path.join(base_dir,"df_divisions2.csv"),index_col=0)
df_segments2=pd.read_csv(path.join(base_dir,"df_segments2.csv"),index_col=[0,"frame","label"])
df_divisions2=pd.read_csv(path.join(base_dir,"df_divisions2.csv"),index_col=0)

segment_labels=df_segments2.xs(0,level="frame",drop_level=False).index.get_level_values("label")
labels=[0]+sorted(list(set(segment_labels.values)))

zarr_path=path.join(base_dir,"image_total_aligned_small2.zarr")
zarr_file=zarr.open(zarr_path,"r")

image=da.from_zarr(zarr_file["image"])
mask=da.from_zarr(zarr_file["mask"])[:,np.newaxis,...]
sizeT=mask.shape[0]
mask[mask==-1]=mask.max().compute()+1

new_segment_id=df_segments2["segment_id"].max()+1
new_index=df_segments2.index.get_level_values(0).max()+1
df_segments3=df_segments2.copy()
for frame in tqdm(range(mask.shape[0])[:]):
    df=df_segments2.xs(frame,level="frame")
    labels=df.index.get_level_values("label").unique()
    labels2=np.unique(mask[frame,0,0]).compute()
    assert np.all(np.isin(labels,labels2))
    noin_labels=labels2[~np.isin(labels2,labels)]
    for noin_label in noin_labels:
        if noin_label>0:
            df_segments3=df_segments3.append(
                pd.Series({
                    "segment_id":new_segment_id,
                },name=(new_index,frame,noin_label))
            )
            new_segment_id=new_segment_id+1
            new_index=new_index+1

df_segments3["segment_id"]=df_segments3["segment_id"].astype(pd.Int32Dtype())

df_segments2=df_segments3
# %%
for frame in tqdm(range(mask.shape[0])[:]):
    reg = regionprops(mask[frame,0,0])
    bboxs = [r.bbox for r in reg]
    labels = [r.label for r in reg]
    for label,bbox in zip(labels,bboxs):
        for j,suffix in enumerate(("y0","x0","y1","x1")):
            df_segments2.loc[(slice(None),frame,label),f"bbox_{suffix}"]=bbox[j]

#%%
df_segments2.to_csv(path.join(base_dir,"df_segments2_updated.csv"))

# %%
