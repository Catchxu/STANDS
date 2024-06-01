import scanpy as sc
import pandas as pd
import numpy as np
from GraphST import GraphST
from GraphST.utils import clustering
import anndata as ad
import torch


def data_preprocess(*args):
    
    adata_list=[]
    
    if not args:
        print("The file path is mandatory!")
        return 
    for item in args:
        try:
            adat = sc.read(item)
            batch = item[item.rfind('/')+1:item.rfind('.h5ad')]
            print(batch)
            adat.obs['batch'] = batch
            adata_list.append(adat)
        except Exception as e:
            raise e
            
    adata = ad.concat(adata_list)
    return adata

def align(adata):
    pos = np.array(adata.obsm['spatial'])
    adata.obs['array_row'] = pos[:, 0]
    adata.obs['array_col'] = pos[:, 1]
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    # define model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GraphST.GraphST(adata, epochs=1200, device=device)
    # train model
    adata = model.train()
    
    return adata

def cluster(adata):
    radius = 50
    n_clusters = adata.obs["label"].nunique()
    print(n_clusters)
    clustering(adata, n_clusters, radius=radius, method='leiden', start=0.1, end=5, increment=0.02, refinement=False)
    return adata


if __name__=="__main__":
    path = []
    raw = data_preprocess(*path)
    adata = align(raw)
    adata = cluster(adata)