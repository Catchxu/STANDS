import dgl
import torch
import numpy as np
import pandas as pd
import networkx as nx
import pyemd
import sys
from functools import partial
from scipy.linalg import toeplitz
from sklearn.neighbors import NearestNeighbors
import itertools
import math
import pulp


def disc(samples1, samples2, kernel, *args, **kwargs):
    samples1 = np.array(samples1)
    samples2 = np.array(samples2)
    d = np.sum(kernel(samples1[:, np.newaxis], samples2[np.newaxis, :], *args, **kwargs))
    return d

def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(float)
    distance_mat = d_mat / distance_scaling
    
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd * emd / (2 * sigma * sigma))

def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):

    # normalize histograms into pmf
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]

    return disc(samples1, samples1, kernel, *args, **kwargs) + \
            disc(samples2, samples2, kernel, *args, **kwargs) - \
            2 * disc(samples1, samples2, kernel, *args, **kwargs)


## get graph stats 

def calculate_degree_histogram(G,node_subset):
    degree = [G.degree(node) for node in node_subset]
    degree_hist = np.bincount(degree,minlength=G.number_of_nodes())
    return degree_hist

def calculate_clustering_coefficient_histogram(G,node_subset,bins=None):
    clustering_coeffs = [nx.clustering(G,node) for node in node_subset]
    hist,_ = np.histogram(clustering_coeffs,bins=bins,range=(0.0,1.0),density=False)
    return hist

def bootstrap_sample(node_set):
    num_samples = len(node_set)
    indices = np.random.choice(num_samples, num_samples, replace=True)
    return [node_set[i] for i in indices]

def calculate_graph_metrics(graph,node_subset,bins = None):
    degree_dist = calculate_degree_histogram(graph,node_subset)
    clustering_dist = calculate_clustering_coefficient_histogram(graph,node_subset,bins=bins)
    return degree_dist,clustering_dist


## process graphs
def dgl_to_nx(dgl_graph, include_classification=True):
    if include_classification:
        g_nx = dgl.to_networkx(dgl_graph.to_simple(), node_attrs=['anomaly', 'classification','position'])
    else:
        g_nx = dgl.to_networkx(dgl_graph.to_simple(), node_attrs=['anomaly','position'])
    return nx.Graph(g_nx)

def get_distributions_for_subsets(predicted_graph, ground_truth_graph, bins=None,num_bootstrap_samples = None):
    results = {}
    

    # predict node
    tp_fp_nodes_pred = [n for n, classif in predicted_graph.nodes(data='classification') if classif in [1, 2]]
    tp_fn_nodes_pred = [n for n, classif in predicted_graph.nodes(data='classification') if classif in [1, 3]]

    # ground truth node
    tp_fp_nodes_gt = []
    for node_pred in tp_fp_nodes_pred:
        position_pred = predicted_graph.nodes[node_pred].get('position')
        node_gt = next((n for n, data in ground_truth_graph.nodes(data=True) if 'position' in data and torch.equal(data.get('position'), position_pred)), None)
        if node_gt is not None:
            tp_fp_nodes_gt.append(node_gt)

    tp_fn_nodes_gt = []
    for node_pred in tp_fn_nodes_pred:
        position_pred = predicted_graph.nodes[node_pred].get('position')
        node_gt = next((n for n, data in ground_truth_graph.nodes(data=True) if 'position' in data and torch.equal(data.get('position'), position_pred)), None)
        if node_gt is not None:
            tp_fn_nodes_gt.append(node_gt)
    
    degree_distributions_pred_tp_fp, clustering_distributions_pred_tp_fp = [], []
    degree_distributions_pred_tp_fn, clustering_distributions_pred_tp_fn = [], []
    degree_distributions_gt_tp_fp, clustering_distributions_gt_tp_fp = [], []
    degree_distributions_gt_tp_fn, clustering_distributions_gt_tp_fn = [], []
    
    for _ in range(num_bootstrap_samples):
        # pred  Bootstrap
        pred_tp_fp_samples = bootstrap_sample(tp_fp_nodes_pred)  #tp fp
        pred_tp_fn_samples = bootstrap_sample(tp_fn_nodes_pred)  #tp fn
        
        # degree dist and cc dist
        
        degree_dist_pred_tp_fp,clustering_dist_pred_tp_fp = calculate_graph_metrics(predicted_graph,pred_tp_fp_samples,bins=bins)
        degree_dist_pred_tp_fn,clustering_dist_pred_tp_fn = calculate_graph_metrics(predicted_graph,pred_tp_fn_samples,bins=bins)
        
        # append to list
        degree_distributions_pred_tp_fp.append(degree_dist_pred_tp_fp)
        clustering_distributions_pred_tp_fp.append(clustering_dist_pred_tp_fp)
        
        degree_distributions_pred_tp_fn.append(degree_dist_pred_tp_fn)
        clustering_distributions_pred_tp_fn.append(clustering_dist_pred_tp_fn)

        # ground truth
        gt_tp_fp_samples = [next((node_gt for node_pred, node_gt in zip(tp_fp_nodes_pred, tp_fp_nodes_gt) if node_pred == sampled_node_pred), None) for sampled_node_pred in pred_tp_fp_samples]
        gt_tp_fn_samples = [next((node_gt for node_pred, node_gt in zip(tp_fn_nodes_pred, tp_fn_nodes_gt) if node_pred == sampled_node_pred), None) for sampled_node_pred in pred_tp_fn_samples]
        
        degree_dist_gt_tp_fp,clustering_dist_gt_tp_fp = calculate_graph_metrics(ground_truth_graph,gt_tp_fp_samples,bins=bins)
        degree_dist_gt_tp_fn,clustering_dist_gt_tp_fn = calculate_graph_metrics(ground_truth_graph,gt_tp_fn_samples,bins=bins)
        
        degree_distributions_gt_tp_fp.append(degree_dist_gt_tp_fp)
        clustering_distributions_gt_tp_fp.append(clustering_dist_gt_tp_fp)
        
        degree_distributions_gt_tp_fn.append(degree_dist_gt_tp_fn)
        clustering_distributions_gt_tp_fn.append(clustering_dist_gt_tp_fn)
        

    results = {
        'Predicted TP+FP Degree': degree_distributions_pred_tp_fp,
        'Predicted TP+FP Clustering': clustering_distributions_pred_tp_fp,
        'Predicted TP+FN Degree': degree_distributions_pred_tp_fn,
        'Predicted TP+FN Clustering': clustering_distributions_pred_tp_fn,
        'Ground Truth TP+FP Degree': degree_distributions_gt_tp_fp,
        'Ground Truth TP+FP Clustering': clustering_distributions_gt_tp_fp,
        'Ground Truth TP+FN Degree': degree_distributions_gt_tp_fn,
        'Ground Truth TP+FN Clustering': clustering_distributions_gt_tp_fn
    }
    
    return results

## label mapping
def solve_assignment_problems(distance_matrix):
    
    problem = pulp.LpProblem("Assignment Problem", pulp.LpMinimize)

    rows = len(distance_matrix)
    cols = len(distance_matrix[0])
    x = [[pulp.LpVariable(f'x_{i}_{j}', cat='Binary') for j in range(cols)] for i in range(rows)]

    # target function
    problem += pulp.lpSum(distance_matrix[i][j] * x[i][j] for i in range(rows) for j in range(cols))

    # constraints 
    for i in range(rows):
        problem += pulp.lpSum(x[i][j] for j in range(cols)) == 1  
        
    for j in range(cols):
        problem += pulp.lpSum(x[i][j] for i in range(rows)) == 1

    # CBC solver 
    solver = pulp.PULP_CBC_CMD()
    problem.solve(solver)

    status = pulp.LpStatus[problem.status]

    minimum_cost = pulp.lpSum(distance_matrix[i][j] * x[i][j].value() for i in range(rows) for j in range(cols))
    assignment_matrix = [[x[i][j].value() for j in range(cols)] for i in range(rows)]
            
    return assignment_matrix

def get_assigned_values(sgd_matrix,assignment_matrix):
    assigned_values = []
    for i,row in enumerate(assignment_matrix):
        for j , assignment in enumerate(row):
            if assignment == 1:
                assigned_value = {
                    f'truth_typeid_{i+1}':i,
                    f'pred_clustid_{j+1}':j,
                    'sgd_value':sgd_matrix[i][j]            
                }
                assigned_values.append(assigned_value)
    return assigned_values

class Build_SGD_graph:
    def __init__(self,adata:ad.AnnData,n_neighbors,spa_key: str = 'spatial'):
        self.adata = adata
        self.position = adata.obsm[spa_key]
        self.n_neighbors = n_neighbors 
        self.typeid_list = sorted(adata.obs['typeid'].unique())
        self.clustid_list = sorted(adata.obs['clustid'].unique())
        
        
    def get_edge(self):
        nbrs = NearestNeighbors(n_neighbors = self.n_neighbors+1)
        nbrs.fit(self.position)   
        _,indices = nbrs.kneighbors(self.position)
        u = indices[:,0].repeat(self.n_neighbors)
        v = indices[:,1:].flatten()
        return u,v
    
    def get_anomaly(self,is_truth = True ,typeid=None,binary_typeid=None):
        if is_truth:
            if typeid == 0:
                anomaly_data = self.adata.obs['typeid'].astype(int)
            else:
                anomaly_data = (self.adata.obs['typeid'] == binary_typeid).astype(int)

        else:
            if typeid == 0:
                anomaly_data = self.adata.obs['clustid'].astype(int)
            else:
                anomaly_data = (self.adata.obs['clustid'] == binary_typeid).astype(int)
            
        return torch.tensor(anomaly_data.values, dtype=torch.float32)

    def classify_cells(self,pred,truth):
        if pred == 1 and truth == 1:
            return 'TP'
        elif pred == 1 and truth == 0:
            return 'FP'
        elif pred == 0 and truth == 1:
            return 'FN'
        else:
            return 'TN'    

    def get_classification(self,pred_graph,truth_graph):
        mapping = {'TP': 1, 'FP': 2, 'FN': 3, 'TN': 4}
        classification_data_list = []
        
        for pred_node,truth_node in zip(pred_graph.ndata['anomaly'],truth_graph.ndata['anomaly']):
            pred = int(pred_node)
            truth = int(truth_node)
            
            classification = self.classify_cells(pred,truth)
            classification_data_list.append(mapping[classification])
        return torch.tensor(classification_data_list,dtype = torch.long)
    
            
    def remove_edges(self, g):
        edges_to_remove = []
        for edge in zip(*g.edges()):
            if g.ndata['anomaly'][edge[0]] != 1 or g.ndata['anomaly'][edge[1]] != 1:
                edges_to_remove.append(edge)
        if edges_to_remove:
            u_list, v_list = zip(*edges_to_remove)
            edge_ids = g.edge_ids(u_list, v_list)
            g.remove_edges(edge_ids)
            
    def build_graph(self):
        u,v = self.get_edge()
        
        predicted_graph_list = []
        g_truth_list = []
        
        for typeid in self.typeid_list:
            if typeid == 0 and len(set(self.typeid_list)) == 2:
                
                #ground truth
                g_truth = dgl.graph((u,v))
                g_truth = dgl.add_self_loop(g_truth)
                g_truth.ndata['anomaly'] =  self.get_anomaly(is_truth = True,typeid = typeid)
                g_truth.ndata['position'] = torch.tensor(self.position,dtype=torch.float32)
                self.remove_edges(g_truth)
                g_truth_list.append(g_truth)
                
                # predict
                predicted_graph = dgl.graph((u,v))
                predicted_graph = dgl.add_self_loop(predicted_graph)
                predicted_graph.ndata['anomaly'] = self.get_anomaly(is_truth = False,typeid = typeid)
                predicted_graph.ndata['position'] = torch.tensor(self.position, dtype=torch.float32)
                
                classification_data = self.get_classification(predicted_graph,g_truth)
                predicted_graph.ndata['classification'] = classification_data
                
                self.remove_edges(predicted_graph)
                predicted_graph_list.append(predicted_graph)
            
            elif typeid == 0 and len(set(self.typeid_list)) != 2:
                continue 
            
            else:
                # ground truth
                g_truth = dgl.graph((u,v))
                g_truth = dgl.add_self_loop(g_truth)
                
                
                g_truth.ndata['anomaly'] = self.get_anomaly(is_truth = True,typeid = typeid,binary_typeid = typeid)
                g_truth.ndata['position'] = torch.tensor(self.position, dtype=torch.float32)
                self.remove_edges(g_truth)
                g_truth_list.append(g_truth)
                
                for clustid in self.clustid_list:
                    if clustid == 0:
                        continue
                    
                    predicted_graph = dgl.graph((u,v))
                    predicted_graph = dgl.add_self_loop(predicted_graph)
                    predicted_graph.ndata['anomaly'] = self.get_anomaly(is_truth = False,typeid = typeid ,binary_typeid = clustid)
                    predicted_graph.ndata['position'] = torch.tensor(self.position, dtype=torch.float32)
                    
                    classification_data = self.get_classification(predicted_graph,g_truth)
                    predicted_graph.ndata['classification'] = classification_data
                    
                    self.remove_edges(predicted_graph)                        
                    predicted_graph_list.append(predicted_graph)
                    
                g_truth_list = list(itertools.chain.from_iterable(itertools.repeat(g_truth,len(set(self.clustid_list))-1) for g_truth in g_truth_list))
                    
        return predicted_graph_list, g_truth_list
    
class SGDEvaluator:
    def __init__(self,adata,n_neighbors,spa_key = 'spatial'):
        self.build_graph_sgd = Build_SGD_graph(adata,n_neighbors,spa_key)
        self.predicted_graph_list = []
        self.g_truth_list = []
    
    def evaluate_sgd(self, predicted_graph_list, g_truth_list, metric='degree'):
        matrix_size = int(math.sqrt(len(predicted_graph_list)))
        sgd_matrix = [[0] * matrix_size for _ in range(matrix_size)]

        for idx,(predicted_graph, g_truth) in enumerate(zip(predicted_graph_list, g_truth_list)):
            predict_nx_graph = dgl_to_nx(predicted_graph, include_classification=True)
            true_nx_graph = dgl_to_nx(g_truth, include_classification=False)

            dist = get_distributions_for_subsets(predict_nx_graph, true_nx_graph,bins = 10,num_bootstrap_samples=50)

            tp_count = torch.sum(predicted_graph.ndata['classification'] == 1).item()
            fn_count = torch.sum(predicted_graph.ndata['classification'] == 3).item()
            weight = tp_count / (tp_count + fn_count)

            if metric == 'degree':
                sgd_metric = 'Degree'
                pred_tp_fp_key = 'Predicted TP+FP Degree'
                gt_tp_fp_key = 'Ground Truth TP+FP Degree'
                pred_tp_fn_key = 'Predicted TP+FN Degree'
                gt_tp_fn_key = 'Ground Truth TP+FN Degree'
            elif metric == 'cc':
                sgd_metric = 'cc'
                pred_tp_fp_key = 'Predicted TP+FP Clustering'
                gt_tp_fp_key = 'Ground Truth TP+FP Clustering'
                pred_tp_fn_key = 'Predicted TP+FN Clustering'
                gt_tp_fn_key = 'Ground Truth TP+FN Clustering'
            else:
                raise ValueError("Invalid metric!")

            # TP&FP
            pred_tp_fp_dist = dist[pred_tp_fp_key]
            gt_tp_fp_dist = dist[gt_tp_fp_key]

            mmd_tp_fp = compute_mmd(gt_tp_fp_dist, pred_tp_fp_dist, kernel=gaussian_emd)

            # TP&FN
            pred_tp_fn_dist = dist[pred_tp_fn_key]
            gt_tp_fn_dist = dist[gt_tp_fn_key]

            mmd_tp_fn = compute_mmd(gt_tp_fn_dist, pred_tp_fn_dist, kernel=gaussian_emd)

            sgd = weight * mmd_tp_fp + (1 - weight) * mmd_tp_fn
            
            # save to matrix
            row = idx // matrix_size
            col = idx % matrix_size
            sgd_matrix[row][col] = sgd
            
        
        if matrix_size == 1:
            return [sgd]
        
        else:
            assignment_matrix = solve_assignment_problems(sgd_matrix)
            
            SGD_list = get_assigned_values(sgd_matrix,assignment_matrix)
            
            return SGD_list