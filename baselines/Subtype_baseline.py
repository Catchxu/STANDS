import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import torch
import dgl
import random

import anndata2ri
from rpy2.robjects import r, globalenv
from rpy2.robjects.packages import importr
import spatialid
from GraphST import GraphST
from GraphST.utils import clustering
import STAGATE
import STAligner
import scipy



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    dgl.random.seed(seed)
    
    
def subtype_camlu_graphst(train: ad.AnnData, test: ad.AnnData, n_cluster: int, random_state: int, device = 'cuda'):
    importr("CAMLU")
    globalenv['seed'] = random_state
    globalenv['train'] = anndata2ri.py2rpy(train)
    globalenv['test'] = anndata2ri.py2rpy(test)
    r("""
    set.seed(seed)

    camlu <- CAMLU(x_train = as.matrix(assay(train,'X')),
                 x_test = as.matrix(assay(test,'X')),
                 ngene=3000, lognormalize=TRUE)
    
    """)

    pre_label = list(r('camlu'))
    test.obs['CAMLU'] = pre_label
    
    seed_everything(random_state)
    adata = test[test.obs['CAMLU']==1]
    model = GraphST.GraphST(adata, device=device, random_seed=random_state)
    adata = model.train()
    clustering(adata, n_clusters = n_cluster, method='leiden', start=0.01, end=1, increment=0.01)
    
    list1 = test.obs['CAMLU'].to_list()
    list2 = adata.obs['leiden'].to_list()
    list2 = [int(i)+1 for i in list2]
    subtype = [list2.pop(0) if i != 0 else 0 for i in list1]
    return subtype


def subtype_scpred_graphst(train: ad.AnnData, test: ad.AnnData, n_cluster: int, random_state: int, device = 'cuda'):
    importr("scPred")
    importr("Seurat")
    importr("magrittr")
    globalenv['seed'] = random_state
    globalenv['train'] = anndata2ri.py2rpy(train)
    globalenv['test'] = anndata2ri.py2rpy(test)
    r("""
    set.seed(seed)
    cell_type <- colData(train)
    train <- CreateSeuratObject(assay(train, 'X'),
                                cell_metadata=colData(train),
                                feat_metadata=rowData(train))    
    train <- train %>%
        NormalizeData() %>%
        FindVariableFeatures() %>%
        ScaleData()
    train = RunPCA(train)
    train@meta.data <- data.frame(train@meta.data, cell_type)

    train <- getFeatureSpace(train, 'cell.type')
    train <- trainModel(train)

    test <- CreateSeuratObject(assay(test, 'X'),
                               cell_metadata=colData(test),
                               feat_metadata=rowData(test))
    test <- NormalizeData(test)
    test <- scPredict(test, train, seed=seed)
    """)
    pre_label = list(r('test@meta.data$scpred_prediction'))
    pre_label = [1 if i == 'unassigned' else 0 for i in pre_label]
    test.obs['scPred'] = pre_label
    
    seed_everything(random_state)
    adata = test[test.obs['scPred']==1]
    model = GraphST.GraphST(adata, device=device, random_seed=random_state)
    adata = model.train()
    clustering(adata, n_clusters = n_cluster, method='leiden', start=0.01, end=1, increment=0.01)
    
    list1 = test.obs['scPred'].to_list()
    list2 = adata.obs['leiden'].to_list()
    list2 = [int(i)+1 for i in list2]
    subtype = [list2.pop(0) if i != 0 else 0 for i in list1]
    return subtype

    
def subtype_chetah_graphst(train: ad.AnnData, test: ad.AnnData, n_cluster: int, random_state: int, device = 'cuda'):
    importr("CHETAH")
    globalenv['seed'] = random_state
    globalenv['train'] = anndata2ri.py2rpy(train)
    globalenv['test'] = anndata2ri.py2rpy(test)
    r("""
    set.seed(seed)

    colnames(colData(train)) <- 'celltypes'
    test <- CHETAHclassifier(input = test, ref_cells = train)
    """)
    pre_label = list(r('colData(test)$celltype_CHETAH'))
    pre_label = [1 if i == 'Unassigned' else 0 for i in pre_label]
    test.obs['CHETAH'] = pre_label
    
    seed_everything(random_state)
    adata = test[test.obs['CHETAH']==1]
    model = GraphST.GraphST(adata, device=device, random_seed=random_state)
    adata = model.train()
    clustering(adata, n_clusters = n_cluster, method='leiden', start=0.01, end=1, increment=0.01)
    
    list1 = test.obs['CHETAH'].to_list()
    list2 = adata.obs['leiden'].to_list()
    list2 = [int(i)+1 for i in list2]
    subtype = [list2.pop(0) if i != 0 else 0 for i in list1]
    return subtype  
    
    
def subtype_scmap_graphst(train: ad.AnnData, test: ad.AnnData, n_cluster: int, random_state: int, device = 'cuda'):
    importr("scmap")
    globalenv['seed'] = random_state
    globalenv['train'] = anndata2ri.py2rpy(train)
    globalenv['test'] = anndata2ri.py2rpy(test)
    r("""
    set.seed(seed)

    logcounts(train) <- assay(train, 'X')
    rowData(train)$feature_symbol <- rownames(train)
    colData(train)$cell_type1 = colData(train)$cell.type
    train <- selectFeatures(train, suppress_plot = TRUE)
    train <- indexCluster(train)

    logcounts(test) <- assay(test, 'X')
    rowData(test)$feature_symbol <- rownames(test)
    scmapCluster_results <- scmapCluster(
      projection = test,
      index_list = list(
        metadata(train)$scmap_cluster_index
      )
    )
    """)

    pre_label = list(r('scmapCluster_results$scmap_cluster_labs'))
    pre_label = [1 if i == 'unassigned' else 0 for i in pre_label]
    test.obs['scmap'] = pre_label
    
    seed_everything(random_state)
    adata = test[test.obs['scmap']==1]
    model = GraphST.GraphST(adata, device=device, random_seed=random_state)
    adata = model.train()
    clustering(adata, n_clusters = n_cluster, method='leiden', start=0.01, end=1, increment=0.01)
    
    list1 = test.obs['scmap'].to_list()
    list2 = adata.obs['leiden'].to_list()
    list2 = [int(i)+1 for i in list2]
    subtype = [list2.pop(0) if i != 0 else 0 for i in list1]
    return subtype  


def subtype_spatialid_stagate(train: ad.AnnData, test: ad.AnnData, n_cluster: int, random_state: int):
    seed_everything(random_state)
    
    spatial = spatialid.transfer.Transfer(spatial_data=test, single_data=train)
    spatial.learn_sc()
    y_score = spatial.annotation()
    
    ratio = 100.0 * (test.obs['anomaly_label'].value_counts()[1] / len(test.obs['anomaly_label']))
    thres = np.percentile(y_score, ratio)
    result = (y_score < thres).astype(int)
    
    test.obs['Spatial-ID'] = result
    adata = test[test.obs['Spatial-ID']==1]
    
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
    STAGATE.Stats_Spatial_Net(adata)
    adata = STAGATE.train_STAGATE(adata, alpha=0)
    sc.pp.neighbors(adata, use_rep='STAGATE')
    sc.tl.umap(adata)
    adata = STAGATE.mclust_R(adata, used_obsm='STAGATE', num_cluster=n_cluster)
    
    list1 = test.obs['spatialid'].to_list()
    list2 = adata.obs['mclust'].to_list()
    list2 = [int(i) for i in list2]
    subtype = [list2.pop(0) if i != 0 else 0 for i in list1]
    return subtype


def subtype_spatialid_staligner_stagate(train: ad.AnnData, test1: ad.AnnData, test2: ad.AnnData, n_cluster: int, random_state: int):
    seed_everything(random_state)

    
    spatial = spatialid.transfer.Transfer(spatial_data=test1, single_data=train)
    spatial.learn_sc()
    y_score = spatial.annotation()
    ratio = 100.0 * (test1.obs['anomaly_label'].value_counts()[1] / len(test1.obs['anomaly_label']))
    thres = np.percentile(y_score, ratio)
    result = (y_score < thres).astype(int)
    test1.obs['Spatial-ID'] = result
    
    spatial = spatialid.transfer.Transfer(spatial_data=test2, single_data=train)
    spatial.learn_sc()
    y_score = spatial.annotation()
    ratio = 100.0 * (test2.obs['anomaly_label'].value_counts()[1] / len(test2.obs['anomaly_label']))
    thres = np.percentile(y_score, ratio)
    result = (y_score < thres).astype(int)
    test2.obs['Spatial-ID'] = result

    test = ad.concat([test1, test2], merge='same')
    adata1 = test1[test1.obs['Spatial-ID']==1]
    adata2 = test2[test2.obs['Spatial-ID']==1]
    adata = test[test.obs['Spatial-ID']==1]
    
    
    Batch_list = []
    adj_list = []
    STAligner.Cal_Spatial_Net(adata1, rad_cutoff=150)
    STAligner.Cal_Spatial_Net(adata2, rad_cutoff=150)

    adj_list.append(adata1.uns['adj'])
    Batch_list.append(adata1)
    adj_list.append(adata2.uns['adj'])
    Batch_list.append(adata2)
    adj_concat = np.asarray(adj_list[0].todense())
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[1].todense()))
    adata.uns['edgeList'] = np.nonzero(adj_concat)
    
    
    adata.obs['batch_name'] = adata.obs['batch']
    adata = STAligner.train_STAligner(adata, verbose=True, knn_neigh = 50, device='cuda')
    
    sid = STAGATE.mclust_R(sid, used_obsm='STAligner', num_cluster=n_cluster)
    
    list1 = test.obs['spatialid'].to_list()
    list2 = adata.obs['mclust'].to_list()
    list2 = [int(i) for i in list2]
    subtype = [list2.pop(0) if i != 0 else 0 for i in list1]
    return subtype


