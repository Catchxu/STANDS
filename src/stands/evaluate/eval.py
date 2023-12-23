import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
from typing import Optional, Literal, Sequence, Union, Tuple
from SGD import Build_SGD_graph, SGDEvaluator



metrics_list = Literal[
    'AUC', 'Precision', 'Recall', 'F1', 'SGD_degree','SGD_cc'
    'ARI', 'NMI', 'ASW_type', '1-ASW_batch', 'BatchKL', 'iLISI', 'cLISI',
]

def evaluate(metrics: Sequence[metrics_list],
             y_true=None, y_score=None, y_pred=None,
             adata: Optional[ad.AnnData]=None,
             batchid: Optional[str]=None, typeid: Optional[str]=None,
             emb: Optional[str] = None, clustid: Optional[str] = None,
             spaid: Optional[str] = None, **kwargs):
    """
    Evaluate performance metrics based on specified evaluation metrics.
    Different metrics require different parameters.
    Here is a description of the metrics that can be calculated and the parameters they require.

    Functions:
        AUC: y_true, y_pred/y_score
        Precision: y_true, y_pred/y_score
        Recall: y_true, y_pred/y_score
        F1: y_true, y_pred/y_score
        SGD_degree: adata, spaid, y_true, y_pred
        SGD_cc: adata, spaid, y_true, y_pred
        ARI: adata, typeid, clustid, (Optional: emb)
        NMI: adata, typeid, clustid, (Optional: emb)
        ASW_type: adata, typeid, (Optional: emb)
        1-ASW_batch: adata, typeid, batchid, (Optional: emb)
        BatchKL: adata, batchid, (Optional: emb)
        iLISI: adata, batchid, (Optional: emb)
        cLISI: adata, typeid, (Optional: emb)

    Parameters:
        metrics (Sequence[str]): List of evaluation metrics to compute.
        y_true (Optional[Union[pd.Series, np.ndarray]]): True labels.
        y_score (Optional[Union[pd.Series, np.ndarray]]): Predicted scores or probabilities.
        y_pred (Optional[Union[pd.Series, np.ndarray]]): Predicted labels.
        adata (Optional[ad.AnnData]): Annotated data containing embeddings or clusters.
        batchid (Optional[str]): Batch ID key in adata.obs for batch information.
        typeid (Optional[str]): Type ID key in adata.obs for type information.
        emb (Optional[str]): Key for embeddings in adata.obsm.
        clustid (Optional[str]): Cluster ID key in adata.obs for clustering information.
        spaid (Optional[str]): Spatial coordinates ID key in adata.obsm (for SGD_degree & SGD_cc metrics).

    Other Parameters:
        n_neighbors (int): Number of neighbors for SGD KNN graph.
        bins (int): Number of equal-width bins in the given range when calculating SGD_cc.
        num_bootstrap_samples (int): Number of bootstrap samples for distribution estimation.
        sigma (int): Sigma parameter for Gaussian Earth Mover's Distance.

    Returns:
        (Union[Tuple, float]): Depending on the number of specified metrics, returns a tuple of metric values or a single metric value.
    
    Note:
        SGD_degree & SGD_cc are available for both anomaly detection and subtyping tasks. 
        They will automatically determine the category based on the types of anomalies in y_true
        eliminating the need for additional parameters to specify whether it is the subtyping task.
    """
    if (y_true is not None) and (y_score is not None):
        y_true = pd.Series(y_true)
        y_score = pd.Series(y_score)

        if y_pred is None:
            ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
            thres = np.percentile(y_score, ratio)
            y_pred = (y_score >= thres).astype(int)
            y_true = y_true.astype(int)

        data = {'y_true': y_true, 'y_score': y_score, 'y_pred': y_pred}

        # for SGD_degree and SGD_cc metrics
        if (adata is not None) and (spaid is not None):
            data.update({'spatial': adata.obsm[spaid]})

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
        if m in ['SGD_degree','SGD_cc']:
            r = method[m](data, **kwargs)
        else:
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
        obs_df = pd.DataFrame(data['type'], columns=['type'])
        meta = ro.conversion.get_conversion().py2rpy(obs_df)
        emb = ro.conversion.get_conversion().py2rpy(data['correct'])

    ro.r('''
    LISI <- function(emb,meta){
        index <- lisi::compute_lisi(emb, meta, c("type"))
        return(median(index$type))
    }
    ''')
    clisi = ro.r['LISI'](emb, meta)
    return np.asarray(clisi)[0]

def eval_SGD_degree(data, **kwargs):
    graph_kwargs = {key: kwargs[key] for key in kwargs if key == 'n_neighbors'}
    eval_kwargs = {key: kwargs[key] for key in kwargs if key != 'n_neighbors'}
    g_pred_list, g_truth_list = Build_SGD_graph(data, **graph_kwargs).build_graph()
    evaluator = SGDEvaluator(**eval_kwargs)
    SGD_degree  = evaluator.evaluate_sgd(g_pred_list, g_truth_list, metric = 'degree')
    return SGD_degree

def eval_SGD_cc(data, **kwargs):
    graph_kwargs = {key: kwargs[key] for key in kwargs if key == 'n_neighbors'}
    eval_kwargs = {key: kwargs[key] for key in kwargs if key != 'n_neighbors'}
    g_pred_list, g_truth_list = Build_SGD_graph(data, **graph_kwargs).build_graph()
    evaluator = SGDEvaluator(**eval_kwargs)
    SGD_cc  = evaluator.evaluate_sgd(g_pred_list, g_truth_list, metric = 'cc')
    return SGD_cc

