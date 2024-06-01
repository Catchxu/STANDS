import scanpy as sc
import pandas as pd
import numpy as np
import scanpy.external as sce
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
            batch = item[item.rfind('/'):item.rfind('.h5ad')+1]
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
    sce.pp.harmony_integrate(adata, key="batch", basis="X_pca",
                         adjusted_basis="X_pca_harmony",
                         max_iter_harmony=15, random_state=0)
    sc.pp.neighbors(adata, use_rep="X_pca_harmony")
    sc.tl.umap(adata)
    return adata

def cluster(adata):
    adata.obsm['emb'] = adata.obsm['X_pca_harmony']
    radius = 50
    n_clusters = adata.obs["cell.type"].nunique()
    print(n_clusters)
    clustering(adata, n_clusters, radius=radius, method='leiden', start=0.1, end=5, increment=0.02, refinement=False)
    return adata


if __name__=="__main__":
    path = []
    raw = data_preprocess(*path)
    adata = align(raw)
    adata = cluster(adata)