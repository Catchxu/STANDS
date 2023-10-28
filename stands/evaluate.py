import scib
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn import metrics
from typing import Optional, Literal, Sequence
from sklearn.neighbors import NearestNeighbors


metrics_list = Literal[
    'AUC', 'Precision', 'Recall', 'F1', 'ARI', 'NMI',
    'ASW_type', '1-ASW_batch', 'BatchKL', 'iLISI', 'cLISI'
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
            data.update({'batch': adata.obs[batchid].values})
        if typeid is not None:
            data.update({'type': adata.obs[typeid].values})
        if clustid is not None:
            data.update({'cluster': adata.obs[clustid].values})
    
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
        'cLISI': eval_cLISI
    }

    result = []
    for m in metrics:
        try:
            r = method[m](data)
        except:
            pass
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
    asw = metrics.silhouette_score(
        X = data['correct'],
        labels = data['batch'],
        metric = 'euclidean'
    )
    asw = (asw + 1)/2
    return 1 - asw

def eval_BatchKL(data, replicates=200, n_neighbors=100, n_cells=100, batch="BatchID"):
    np.random.seed(1)

    eval_data = data['correct']
    batch = data['batch']
    table_batch = np.bincount(batch)
    tmp00 = table_batch / np.sum(table_batch)
    n = eval_data.shape[0]

    KL = []
    for _ in range(replicates):
        bootsamples = np.random.choice(n, n_cells)
        nn = NearestNeighbors(n_neighbors=min(5 * len(tmp00), n_neighbors)).fit(eval_data)
        _, indices = nn.kneighbors(eval_data[bootsamples, :])
    
        KL_x = []
        for y in range(len(bootsamples)):
            id = indices[y]
            tmp = np.bincount(eval_data[id])
            tmp = tmp / np.sum(tmp)
            KL_x.append(np.sum(tmp * np.log2(tmp / tmp00), where=tmp > 0))
        
        KL.append(np.mean(KL_x))

    return np.mean(np.array(KL))

def eval_iLISI(data):
    obs_df = pd.DataFrame(data['batch'], columns='batch')
    adata = ad.AnnData(X=data['correct'], obs=obs_df)
    return scib.metrics.ilisi_graph(adata, 'batch', 'knn', use_rep='X', scale=False)

def eval_cLISI(data):
    obs_df = pd.DataFrame(data['type'], columns='celltype')
    adata = ad.AnnData(X=data['correct'], obs=obs_df)
    return scib.metrics.ilisi_graph(adata, 'celltype', 'knn', use_rep='X', scale=False)




if __name__=='main':
    y_true = [0, 1, 1]
    y_score = [0.1, 0.9, 0.8]
    evaluate('AUC', y_true, y_score)
    evaluate(['AUC', 'F1'], y_true, y_score)