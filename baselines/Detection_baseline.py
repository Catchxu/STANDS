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



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    dgl.random.seed(seed)


def detect_scpred(train: ad.AnnData, test: ad.AnnData, random_state: int):
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
    result = {'cell_type': test.obs['cell.type'].values,
              'diff': pre_label}
    result = pd.DataFrame(result, index=test.obs.index)
    return result


def detect_chetah(train: ad.AnnData, test: ad.AnnData, random_state: int):
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
    result = {'cell_type': test.obs['cell.type'].values,
              'diff': pre_label}
    result = pd.DataFrame(result, index=test.obs.index)
    return result


def detect_scmap(train: ad.AnnData, test: ad.AnnData, random_state: int):
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
    result = {'cell_type': test.obs['cell.type'].values,
              'diff': pre_label}
    result = pd.DataFrame(result, index=test.obs.index)
    return result


def detect_camlu(train: ad.AnnData, test: ad.AnnData, random_state: int):
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
    result = {'cell_type': test.obs['cell.type'].values,
              'diff': pre_label}
    result = pd.DataFrame(result, index=test.obs.index)
    return result


def detect_spatialid(train: ad.AnnData, test: ad.AnnData, random_state: int):
    seed_everything(random_state)
    
    spatial = spatialid.transfer.Transfer(spatial_data=test, single_data=train)
    spatial.learn_sc()
    y_score = spatial.annotation()
    
    ratio = 100.0 * (test.obs['anomaly_label'].value_counts()[1] / len(test.obs['anomaly_label']))
    thres = np.percentile(y_score, ratio)
    result = (y_score < thres).astype(int)
    return result