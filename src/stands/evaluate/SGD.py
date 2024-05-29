import dgl
import torch
import numpy as np
import networkx as nx
import pyemd
from scipy.linalg import toeplitz
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wasserstein_distance
import itertools
import math
import pulp
import anndata as ad
from typing import Dict, Tuple, Optional, List, Literal


def disc(samples1, samples2, kernel, *args, **kwargs):
    samples1 = np.array(samples1)
    samples2 = np.array(samples2)
    d = np.mean(kernel(samples1, samples2, *args, **kwargs))
    return d

def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    support_size = x.shape[1]
    d_mat = toeplitz(range(support_size)).astype(float)
    distance_mat = d_mat / distance_scaling
    emd = np.zeros((x.shape[0], y.shape[0]))

    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            emd[i, j] = pyemd.emd(x[i], y[j], distance_mat) * 2
    return np.exp(-emd / (2 * sigma * sigma))

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
    degree_hist = np.bincount(degree, minlength=G.number_of_nodes())
    return degree_hist

def calculate_clustering_coefficient_histogram(G, node_subset, bins=None):
    clustering_coeffs = [nx.clustering(G, node) for node in node_subset]
    hist,_ = np.histogram(clustering_coeffs, bins=bins, range=(0.0,1.0), density=False)
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




class Build_SGD_graph:
    """
    Spatial locations are represented as nodes in an undirected graph.Normal spots are isolated, 
    while anomalous spots are connected to their k-nearest anomalous neighbors. Note that in the 
    anomaly detection results, incorrectly identified spots as anomalies (false positives) become 
    connected, and false negatives become isolated, which leads to a deviation from the local 
    structures of the ground truth graph. Spots are divided into two regions: one includes true 
    positives plus false positives (TP+FP) anomalies, and the other includes true positives plus 
    false negatives (TP+FN) anomalies. We perform a bootstrap sampling of m sets of spots from these 
    two regions.

    Examples:
        >>> g_pred_list, g_truth_list = Build_SGD_graph(data, n_neighbors = 6).build_graph()
    """
    def __init__(self, data: Dict, n_neighbors: str = 6):
        """
        Initialize the Build_SGD_graph class. 

        Parameters:
            data (Dict): Dictionary containing spatial data and labels.
            n_neighbors (int): Number of neighbors to consider for edge creation (default is 6).
        """
        self.data = data
        self.position = data['spatial']
        self.n_neighbors = n_neighbors
        self.truth_list = np.delete(np.unique(data['y_true']), 0)
        self.pred_list = np.unique(data['y_pred'])

    def get_edge(self):
        """
        Generate edges based on spatial positions.

        Returns:
            (Tuple[np.ndarray]): Two arrays representing source and destination nodes for edges.
        """
        nbrs = NearestNeighbors(n_neighbors = self.n_neighbors + 1)
        nbrs.fit(self.position)   
        _, indices = nbrs.kneighbors(self.position)
        u = indices[:,0].repeat(self.n_neighbors)
        v = indices[:,1:].flatten()
        return u, v

    def get_anomaly(self, is_truth=True, typeid=None, binary_typeid=None):
        """
        Extract anomaly information for nodes.

        Parameters:
            is_truth (bool): If True, extract anomaly data from ground truth labels. If False, extract from predicted labels.
            typeid (Optional[int]): Type ID for filtering data.
            binary_typeid (Optional[int]): Binary Type ID for filtering data.

        Returns:
            (torch.Tensor): Anomaly data as a tensor.
        """
        if is_truth:
            if typeid == 0:
                anomaly_data = np.array(self.data['y_true']).astype(int)
            else:
                anomaly_data = np.array(self.data['y_true'] == binary_typeid).astype(int)

        else:
            if typeid == 0:
                anomaly_data = np.array(self.data['y_pred']).astype(int)
            else:
                anomaly_data = np.array(self.data['y_pred'] == binary_typeid).astype(int)
            
        return torch.tensor(anomaly_data, dtype=torch.float32)

    def classify_cells(self, pred: int, truth: int):
        """
        Classify cells based on prediction and ground truth.

        Parameters:
            pred (int): Predicted flag (1 for anomaly, 0 for normal).
            truth (int): Ground truth flag (1 for anomaly, 0 for normal).

        Returns:
            (str): Classification result (TP, FP, FN, or TN).
        """
        if pred == 1 and truth == 1:
            return 'TP'
        elif pred == 1 and truth == 0:
            return 'FP'
        elif pred == 0 and truth == 1:
            return 'FN'
        else:
            return 'TN'    

    def get_classification(self, pred_graph: dgl.DGLGraph, truth_graph: dgl.DGLGraph):
        """
        Get classification labels for nodes in predicted and ground truth graphs.

        Parameters:
            pred_graph (dgl.DGLGraph): Predicted graph.
            truth_graph (dgl.DGLGraph): Ground truth graph.

        Returns:
            (torch.Tensor): Classification labels as a tensor.
        """
        mapping = {'TP': 1, 'FP': 2, 'FN': 3, 'TN': 4}
        classification_data_list = []
        
        for pred_node, truth_node in zip(pred_graph.ndata['anomaly'], truth_graph.ndata['anomaly']):
            pred = int(pred_node)
            truth = int(truth_node)
            
            classification = self.classify_cells(pred, truth)
            classification_data_list.append(mapping[classification])
        return torch.tensor(classification_data_list,dtype = torch.long)
    
    def remove_edges(self, g: dgl.DGLGraph):
        """
        Remove edges from the graph based on anomaly information.

        Parameters:
            g (dgl.DGLGraph): Input graph.
        """
        edges_to_remove = []
        for edge in zip(*g.edges()):
            if g.ndata['anomaly'][edge[0]] != 1 or g.ndata['anomaly'][edge[1]] != 1:
                edges_to_remove.append(edge)
        if edges_to_remove:
            u_list, v_list = zip(*edges_to_remove)
            edge_ids = g.edge_ids(u_list, v_list)
            g.remove_edges(edge_ids)
            
    def build_graph(self):
        """
        Build graphs for predicted and ground truth labels.

        Returns:
            (List[dgl.DGLGraph]): List of DGLGraph objects representing the predicted and ground truth graphs.
        """
        u, v = self.get_edge()
        
        g_pred_list = []
        g_truth_list = []
        
        for typeid in self.truth_list:
            if typeid == 0 and len(self.truth_list) == 2:
                #ground truth
                g_truth = dgl.graph((u,v))
                g_truth = dgl.add_self_loop(g_truth)
                g_truth.ndata['anomaly'] =  self.get_anomaly(is_truth=True, typeid=typeid)
                g_truth.ndata['position'] = torch.tensor(self.position, dtype=torch.float32)
                self.remove_edges(g_truth)
                g_truth_list.append(g_truth)
                
                # predict
                g_pred = dgl.graph((u,v))
                g_pred = dgl.add_self_loop(g_pred)
                g_pred.ndata['anomaly'] = self.get_anomaly(is_truth=False, typeid=typeid)
                g_pred.ndata['position'] = torch.tensor(self.position, dtype=torch.float32)
                
                classification_data = self.get_classification(g_pred, g_truth)
                g_pred.ndata['classification'] = classification_data
                
                self.remove_edges(g_pred)
                g_pred_list.append(g_pred)
            
            elif typeid == 0 and len(set(self.truth_list)) != 2:
                continue 
            
            else:
                # ground truth
                g_truth = dgl.graph((u,v))
                g_truth = dgl.add_self_loop(g_truth)
                
                g_truth.ndata['anomaly'] = self.get_anomaly(True, typeid, typeid)
                g_truth.ndata['position'] = torch.tensor(self.position, dtype=torch.float32)
                self.remove_edges(g_truth)
                g_truth_list.append(g_truth)
                
                for predid in self.pred_list:
                    if predid == 0:
                        continue
                    
                    g_pred = dgl.graph((u,v))
                    g_pred = dgl.add_self_loop(g_pred)
                    g_pred.ndata['anomaly'] = self.get_anomaly(False, typeid, predid)
                    g_pred.ndata['position'] = torch.tensor(self.position, dtype=torch.float32)
                    
                    classification_data = self.get_classification(g_pred,g_truth)
                    g_pred.ndata['classification'] = classification_data
                    
                    self.remove_edges(g_pred)                        
                    g_pred_list.append(g_pred)
                    
                g_truth_list = list(itertools.chain.from_iterable(itertools.repeat(g_truth,len(set(self.pred_list))-1) for g_truth in g_truth_list))
        
        return g_pred_list, g_truth_list




class SGDEvaluator:
    """
    Evaluate SGD_degree or SGD_cc with the built predicted and ground truth SGD Graph.

    Examples:
        >>> evaluator = SGDEvaluator(bins = 10, num_bootstrap_samples = 50, sigma = 1)
        >>> evaluator.evaluate_sgd(g_pred_list, g_truth_list, metric = 'degree')
        1.01
        >>> evaluator.evaluate_sgd(g_pred_list, g_truth_list, metric = 'cc')
        0.34
    """
    def __init__(self, bins: int = 10, num_bootstrap_samples: int = 10, sigma: int = 1):
        """
        Initialize the SGDEvaluator.

        Parameters:
            bins (int): Number of equal-width bins in the given range when calculating SGD_cc.
            num_bootstrap_samples (int): Number of bootstrap samples for distribution estimation.
            sigma (int): Sigma parameter for Gaussian Earth Mover's Distance.
        """
        self.g_pred_list = []
        self.g_truth_list = []
        self.bins = bins
        self.num_bootstrap_samples = num_bootstrap_samples
        self.sigma = sigma
    
    def evaluate_sgd(self, g_pred_list: List[dgl.DGLGraph],
                     g_truth_list: List[dgl.DGLGraph],
                     metric: Literal['degree', 'cc']):
        """
        Evaluate SGD based on predicted and ground truth graphs.

        Parameters:
            g_pred_list (List[dgl.DGLGraph]): List of predicted DGLGraphs.
            g_truth_list (List[dgl.DGLGraph]): List of ground truth DGLGraphs.
            metric (Literal['degree', 'cc']): Metric to evaluate ('degree' or 'cc').

        Returns:
            (float): SGD score for the predicted and ground truth graphs.
        """
        matrix_size = int(math.sqrt(len(g_pred_list)))
        sgd_matrix = [[0] * matrix_size for _ in range(matrix_size)]

        for idx,(g_pred, g_truth) in enumerate(zip(g_pred_list, g_truth_list)):
            pred_nx_graph = dgl_to_nx(g_pred, include_classification=True)
            true_nx_graph = dgl_to_nx(g_truth, include_classification=False)

            dist = get_distributions_for_subsets(pred_nx_graph, true_nx_graph,
                                                 self.bins, self.num_bootstrap_samples)

            tp_count = torch.sum(g_pred.ndata['classification'] == 1).item()
            fn_count = torch.sum(g_pred.ndata['classification'] == 3).item()
            weight = tp_count / (tp_count + fn_count)

            if metric == 'degree':
                pred_tp_fp_key = 'Predicted TP+FP Degree'
                gt_tp_fp_key = 'Ground Truth TP+FP Degree'
                pred_tp_fn_key = 'Predicted TP+FN Degree'
                gt_tp_fn_key = 'Ground Truth TP+FN Degree'
            elif metric == 'cc':
                pred_tp_fp_key = 'Predicted TP+FP Clustering'
                gt_tp_fp_key = 'Ground Truth TP+FP Clustering'
                pred_tp_fn_key = 'Predicted TP+FN Clustering'
                gt_tp_fn_key = 'Ground Truth TP+FN Clustering'
            else:
                raise ValueError("Invalid metric!")

            # TP&FP
            pred_tp_fp_dist = dist[pred_tp_fp_key]
            gt_tp_fp_dist = dist[gt_tp_fp_key]

            mmd_tp_fp = compute_mmd(gt_tp_fp_dist, pred_tp_fp_dist, kernel=gaussian_emd, sigma=self.sigma)

            # TP&FN
            pred_tp_fn_dist = dist[pred_tp_fn_key]
            gt_tp_fn_dist = dist[gt_tp_fn_key]

            mmd_tp_fn = compute_mmd(gt_tp_fn_dist, pred_tp_fn_dist, kernel=gaussian_emd, sigma=self.sigma)

            sgd = weight * mmd_tp_fp + (1 - weight) * mmd_tp_fn
            
            # save to matrix
            row = idx // matrix_size
            col = idx % matrix_size
            sgd_matrix[row][col] = sgd


        if matrix_size == 1:
            return sgd
        else:
            assignment_matrix = self.solve_assignment_problems(sgd_matrix)
            SGD_list = self.get_assigned_values(sgd_matrix, assignment_matrix)
            return np.average(SGD_list)

    def solve_assignment_problems(self, distance_matrix: List[List[float]]):
        """
        Solve the Assignment Problem using the CBC solver.

        Parameters:
            distance_matrix (List[List[float]]): Matrix representing the costs of assignments (distance).

        Returns:
            (List[List[int]]): Assignment matrix indicating optimal assignments.
        """
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
    
    
    def get_assigned_values(self, sgd_matrix: List[List[float]], assignment_matrix: List[List[int]]):
        """
        Get assigned values from the SGD matrix and assignment matrix.

        Parameters:
            sgd_matrix (List[List[float]]): Matrix of SGD values.
            assignment_matrix (List[List[int]]): Assignment matrix indicating optimal assignments.

        Returns:
            (List[float]): List of dictionaries containing assigned values.
        """
        assigned_values = []
        for i,row in enumerate(assignment_matrix):
            for j, assignment in enumerate(row):
                if assignment == 1:
                    assigned_values.append(sgd_matrix[i][j])
        return assigned_values