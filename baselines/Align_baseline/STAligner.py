import numpy as np
import scanpy as sc
import anndata as ad
import STAligner
import scipy.sparse as sp
import scipy.linalg
import torch


def data_preprocess(*args):
    
    adata_list=[]
    adj_list = []
    
    if not args:
        print("The file path is mandatory!")
        return 
    for item in args:
        try:
            adat = sc.read(item)
            STAligner.Cal_Spatial_Net(adat, rad_cutoff=500)
            sc.pp.normalize_total(adat, target_sum=1e4)
            batch = item[item.rfind('/')+1:item.rfind('.h5ad')]
            adat.obs['batch'] = batch
            adata_list.append(adat)
            adj_list.append(data.uns['adj'])
        except Exception as e:
            raise e
            
    adata_concat = ad.concat(adata_list)
    adata_concat.obs_names_make_unique()
    for batch_id in range(1,3):
        adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
        adata_concat.uns['edgeList'] = np.nonzero(adj_concat)
    return adata

def align(adata):
    pos = np.array(adata.obsm['spatial'])
    adata.obs['array_row'] = pos[:, 0]
    adata.obs['array_col'] = pos[:, 1]
    used_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 50, device=used_device)
    sc.pp.neighbors(adata, use_rep='STAligner')
    sc.tl.umap(adata)
    return adata

def cluster(adata):
    adata.obsm['emb'] = adata.obsm['STAligner']
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