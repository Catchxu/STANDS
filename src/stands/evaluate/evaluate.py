import scib
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
from typing import Optional, Literal, Sequence
from SGD import *



metrics_list = Literal[
    'AUC', 'Precision', 'Recall', 'F1', 'ARI', 'NMI',
    'ASW_type', '1-ASW_batch', 'BatchKL', 'iLISI', 'cLISI','SGD_degree','SGD_cc'
]

def evaluate(metrics: Sequence[metrics_list],
             y_true=None, y_score=None, adata: Optional[ad.AnnData]=None,
             batchid: Optional[str]=None, typeid: Optional[str]=None,
             emb: Optional[str] = None, clustid: Optional[str] = None):
    """calculate evaluation metrics"""
    if (y_true is not None) and(y_score is not None):
        y_true = pd.Series(y_true)
        y_score = pd.Series(y_score)

        ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
        thres = np.percentile(y_score, ratio)
        y_pred = (y_score >= thres).astype(int)
        y_true = y_true.astype(int)

        data = {'y_true': y_true, 'y_score': y_score, 'y_pred': y_pred}
    
    elif adata is not None:
        if emb is None:
            sc.tl.tsne(adata, random_state=0, use_fast_tsne=False)
            correct = adata.obsm['X_tsne']
        else:
            correct = adata.obsm[emb]

        data = {'correct': correct}
        
        if batchid is not None:
            _, idx = np.unique(adata.obs[batchid].values, return_inverse=True)
            data.update({'batch': idx})
        if typeid is not None:
            _, idx = np.unique(adata.obs[typeid].values, return_inverse=True)
            data.update({'type': idx})
        if clustid is not None:
            _, idx = np.unique(adata.obs[clustid].values, return_inverse=True)
            data.update({'cluster': idx})

    method = {
        'AUC': eval_AUC,
        'Precision': eval_P, 
        'Recall': eval_R, 
        'F1': eval_F1,
        'ARI': eval_ARI,
        'NMI': eval_NMI,
        'ASW_type': eval_ASW_type,
        '1-ASW_batch': eval_ASW_batch,
        'BatchKL': eval_BatchKL,
        'iLISI': eval_iLISI,
        'cLISI': eval_cLISI,
        'SGD_degree': eval_SGD_degree,
        'SGD_cc': eval_SGD_cc
    }

    result = []
    for m in metrics:
        r = method[m](data)
        result.append(r)

    if len(result) >= 2:
        return tuple(result)
    else:
        return result[0]

def eval_AUC(data):
    return metrics.roc_auc_score(data['y_true'], data['y_score'])

def eval_P(data):
    return metrics.precision_score(data['y_true'], data['y_pred'])

def eval_R(data):
    return metrics.recall_score(data['y_true'], data['y_pred'])

def eval_F1(data):
    return metrics.f1_score(data['y_true'], data['y_pred'], average='binary')

def eval_ARI(data):
    return metrics.adjusted_rand_score(data['type'], data['cluster'])

def eval_NMI(data):
    return metrics.normalized_mutual_info_score(data['type'], data['cluster'])

def eval_ASW_type(data):
    asw = metrics.silhouette_score(
        X = data['correct'], 
        labels = data['type'],
        metric = 'euclidean'
    )
    asw = (asw + 1)/2
    return asw

def eval_ASW_batch(data):
    obs_df = {key: value for key, value in data.items() if key in ['type', 'batch']}
    adata = ad.AnnData(np.array(data['correct'], dtype=np.float32), obs=obs_df)
    group = adata.obs['type'].unique()
    asw = []
    for g in group:
        subadata = adata[adata.obs['type'] == g, :]
        if len(subadata.obs['batch'].unique()) == 1:
            continue
        s = metrics.silhouette_score(
            X = subadata.X, 
            labels = subadata.obs['batch'].values,
            metric = 'euclidean'
            )
        s = 1 - np.abs(s)
        asw.append(s)
    return np.mean(np.array(asw))

def eval_BatchKL(data, replicates=200, n_neighbors=100, n_cells=100):
    np.random.seed(1)

    eval_data = data['correct']
    batch = data['batch']
    table_batch = np.bincount(batch)
    q = table_batch / np.sum(table_batch)
    n = eval_data.shape[0]

    KL = []
    for _ in range(replicates):
        bootsamples = np.random.choice(n, n_cells)
        nn = NearestNeighbors(n_neighbors=min(5 * len(q), n_neighbors)).fit(eval_data)
        _, indices = nn.kneighbors(eval_data[bootsamples, :])

        KL_x = []
        for y in range(len(bootsamples)):
            id = indices[y]
            p = np.bincount(batch[id])
            p = p / np.sum(p)

            if len(p) < len(q):
                delta = len(q) - len(p)
                p = np.append(p, [0]*delta)

            with np.errstate(divide='ignore', invalid='ignore'):
                KL_x.append(np.sum(p*np.log2(p/q), where=p>0))
        
        KL.append(np.mean(KL_x))

    return np.mean(np.array(KL))

def eval_iLISI(data):
    with localconverter(pandas2ri.converter + numpy2ri.converter):
        obs_df = pd.DataFrame(data['batch'], columns=['batch'])
        meta = ro.conversion.get_conversion().py2rpy(obs_df)
        emb = ro.conversion.get_conversion().py2rpy(data['correct'])

    ro.r('''
    LISI <- function(emb,meta){
        index <- lisi::compute_lisi(emb, meta, c("batch"))
        return(median(index$batch))
    }
    ''')
    ilisi = ro.r['LISI'](emb, meta)
    return np.asarray(ilisi)[0]

def eval_cLISI(data):
    with localconverter(pandas2ri.converter + numpy2ri.converter):
        obs_df = pd.DataFrame(data['celltype'], columns=['celltype'])
        meta = ro.conversion.get_conversion().py2rpy(obs_df)
        emb = ro.conversion.get_conversion().py2rpy(data['correct'])

    ro.r('''
    LISI <- function(emb,meta){
        index <- lisi::compute_lisi(emb, meta, c("celltype"))
        return(median(index$celltype))
    }
    ''')
    clisi = ro.r['LISI'](emb, meta)
    return np.asarray(clisi)[0]

def eval_SGD_degree(adata):
    g_pred_list,g_truth_list = Build_SGD_graph(adata,n_neighbors = 6,spa_key = 'spatial')

    evaluator = SGDEvaluator(adata,n_neighbors = 6,spa_key = 'spatial')

    SGD_degree  = evaluator.evaluate_sgd(g_pred_list,g_truth_list,metric = 'degree')
    
    return SGD_degree
    
    
def eval_SGD_cc(adata):
    g_pred_list,g_truth_list = Build_SGD_graph(adata,n_neighbors = 6,spa_key = 'spatial')

    evaluator = SGDEvaluator(adata,n_neighbors = 6,spa_key = 'spatial')

    SGD_degree  = evaluator.evaluate_sgd(g_pred_list,g_truth_list,metric = 'cc')
    
    return SGD_cc

