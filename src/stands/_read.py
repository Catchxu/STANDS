import warnings
import numpy as np
import scanpy as sc
import anndata as ad
from math import e
from typing import Literal, Optional, List

from ._utils import seed_everything, clear_warnings
from ._graph import BuildGraph, BuildMultiGraph


def preprocess_data(adata: ad.AnnData):
    # clear the obs &var names
    adata = adata[:, adata.var_names.notnull()]
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # prefilter special genes
    drop_pattern1 = adata.var_names.str.startswith('ERCC')
    drop_pattern2 = adata.var_names.str.startswith('MT-')
    drop_pattern = np.logical_and(~drop_pattern1, ~drop_pattern2)
    adata._inplace_subset_var(drop_pattern)

    # normalization
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata, base=e)
    return adata


def set_patch(adata: ad.AnnData):
    try:
        info = next(iter(adata.uns['spatial'].values()))['scalefactors']
        patch_size = info['fiducial_diameter_fullres']*info['tissue_hires_scalef']
        patch_size = int(patch_size)
        n = np.ceil(np.log(patch_size)/np.log(2)).astype(int)
        patch_size = 2**n
    except:
        patch_size = 32
    return patch_size


@clear_warnings
def read(adata: ad.AnnData, preprocess: bool = True,
         return_type: Literal['anndata', 'graph', 'tuple'] = 'graph',
         n_genes: int = 3000, n_neighbors: int = 4, spa_key: str = 'spatial',
         patch_size: Optional[int] = None, **kwargs):
    """
    Read single spatial data and preprocess if required.
    The read data are transformed to one graph.

    Parameters:
        adata (ad.AnnData): AnnData object.
        spa_key (str): Key for spatial information in AnnData objects.
        preprocess (bool): Perform data preprocessing.
        n_genes (int): Number of genes for feature selection.
        patch_size (Optional[int]): Patch size for H&E images.
        n_neighbors (int): Number of neighbors for spatial data reading.
        return_type (Literal['anndata', 'graph', 'tuple']): Type of data to return.

    Other Parameters:
        train_mode (bool): Whether to use train mode with data augmentation.

    Returns:
        (Union[ad.AnnData, Tuple, Dict]): Depending on the 'return_type', returns either a tuple of AnnData objects or a dictionary of graph-related data.
    """
    seed_everything(0)

    position = adata.obsm[spa_key]

    try:
        image = next(iter(adata.uns['spatial'].values()))['images']['hires']
    except:
        image = None

    if preprocess:
        adata = preprocess_data(adata)
        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_genes, subset=True)

    if return_type == 'anndata':
        return adata

    elif return_type == 'tuple':
        return adata, image, position

    elif return_type == 'graph':
        if patch_size is None:
            patch_size = set_patch(adata)
        builder = BuildGraph(adata, image, position,
                             n_neighbors=n_neighbors, patch_size=patch_size, **kwargs)
        return builder.pack()


@clear_warnings
def read_cross(ref: ad.AnnData, tgt: ad.AnnData, spa_key: str = 'spatial',
               preprocess: bool = True, n_genes: int = 3000, patch_size: Optional[int] = None,
               n_neighbors: int = 4, return_type: Literal['anndata', 'graph'] = 'graph', **kwargs):
    """
    Read spatial data from two sources and preprocess if required.
    The read data are transformed to reference and target graph.

    Parameters:
        ref (ad.AnnData): Reference AnnData object.
        tgt (ad.AnnData): Target AnnData object.
        spa_key (str): Key for spatial information in AnnData objects.
        preprocess (bool): Perform data preprocessing.
        n_genes (int): Number of genes for feature selection.
        patch_size (Optional[int]): Patch size for H&E images.
        n_neighbors (int): Number of neighbors for spatial data reading.
        return_type (Literal['anndata', 'graph']): Type of data to return.

    Other Parameters:
        train_mode (bool): Whether to use train mode with data augmentation.

    Returns:
        (Union[Tuple, Dict]): Depending on the 'return_type', returns either a tuple of AnnData objects or a dictionary of graph-related data.
    """
    seed_everything(0)

    ref, ref_img, ref_pos = read(ref, False, 'tuple', spa_key=spa_key, n_neighbors=n_neighbors)
    tgt, tgt_img, tgt_pos = read(tgt, False, 'tuple', spa_key=spa_key, n_neighbors=n_neighbors)
    overlap_gene = list(set(ref.var_names) & set(tgt.var_names))
    ref = ref[:, overlap_gene]
    tgt = tgt[:, overlap_gene]

    if preprocess:
        ref = preprocess_data(ref)
        tgt = preprocess_data(tgt)
        if len(overlap_gene) <= n_genes:
            warnings.warn(
                'There are too few overlapping genes to perform feature selection'
            )
        else:
            sc.pp.filter_genes(ref, min_cells=10)
            sc.pp.highly_variable_genes(ref, n_top_genes=n_genes, subset=True)
            tgt = tgt[:, ref.var_names]
    
    if return_type == 'anndata':
        return ref, tgt

    elif return_type == 'graph':
        if patch_size is None:
            patch_size = set_patch(ref)

        ref_b = BuildGraph(ref, ref_img, ref_pos,
                           n_neighbors=n_neighbors, patch_size=patch_size, **kwargs)
        tgt_b = BuildGraph(tgt, tgt_img, tgt_pos,
                           n_neighbors=n_neighbors, patch_size=patch_size, **kwargs)
        return ref_b.pack(), tgt_b.pack()


@clear_warnings
def read_multi(adata_list: List[ad.AnnData], patch_size: Optional[int] = None,
               gene_list: Optional[List[str]] = None, preprocess: bool = True, 
               n_genes: int = 3000, n_neighbors: int = 4,
               return_type: Literal['anndata', 'graph'] = 'graph', 
               spa_key: str = 'spatial', **kwargs):
    """
    Read multiple spatial datasets and preprocess if required.
    All the datasets are transformed to only one graph.

    Parameters:
        adata_list (List[ad.AnnData]): List of AnnData objects.
        patch_size (Optional[int]): Patch size for H&E images.
        gene_list (Optional[List[str]]): Selected gene list.
        preprocess (bool): Perform data preprocessing.
        n_genes (int): Number of genes for feature selection.
        n_neighbors (int): Number of neighbors for spatial data reading.
        return_type (Literal['anndata', 'graph']): Type of data to return.
        spa_key (str): Key for spatial information in AnnData objects.
    
    Other Parameters:
        train_mode (bool): Whether to use train mode with data augmentation.

    Returns:
        (Union[List, Dict]): Depending on the 'return_type', returns either a list of AnnData objects or a dictionary of graph-related data.
    """
    seed_everything(0)

    adatas, images, positions = [], [], []
    for i in range(len(adata_list)):
        d, img, pos = read(adata_list[i], False, 'tuple', spa_key=spa_key, n_neighbors=n_neighbors)
        adatas.append(d)
        images.append(img)
        positions.append(pos)

    for img in images:
        if img is None:
            images = None
            break

    if preprocess:
        adatas = [preprocess_data(d) for d in adatas]
        if gene_list is None:
            ref = adatas[0]
            sc.pp.filter_genes(ref, min_cells=10)
            sc.pp.highly_variable_genes(ref, n_top_genes=n_genes, subset=True)
            adatas = [d[:, list(ref.var_names)] for d in adatas]
        else:
            adatas = [d[:, list(gene_list)] for d in adatas]

    if return_type == 'anndata':
        return adatas

    elif return_type == 'graph':
        if patch_size is None:
            patch_size = set_patch(adatas[0])
        builder = BuildMultiGraph(adatas, images, positions, 
                                  n_neighbors=n_neighbors, patch_size=patch_size, **kwargs)
        return builder.pack()


