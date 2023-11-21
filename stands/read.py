import warnings
import numpy as np
import scanpy as sc
import anndata as ad
from math import e
from typing import Literal, Optional, List

from ._utils import seed_everything
from ._graph import Build_graph, Build_multi_graph


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


def read(data_dir: Optional[str] = None, data_name: Optional[str] = None,
         adata: Optional[ad.AnnData] = None, preprocess: bool = True,
         return_type: Literal['anndata', 'graph', 'multi'] = 'graph',
         n_genes: int = 3000, n_neighbors: int = 4, spa_key: str = 'spatial',
         patch_size: Optional[int] = None, train_mode: bool = True):
    seed_everything(0)

    if adata is None:
        if (data_dir is None) or (data_name is None):
            RuntimeError('Please set the read file/path.')
        input_dir = data_dir + data_name + '.h5ad'
        adata = sc.read_h5ad(input_dir)

    position = adata.obsm[spa_key]

    try:
        image = next(iter(adata.uns['spatial'].values()))['images']['hires']
    except:
        image = None

    if preprocess:
        adata = preprocess_data(adata)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            sc.pp.filter_genes(adata, min_cells=10)
            sc.pp.highly_variable_genes(adata, n_top_genes=n_genes, subset=True)

    if return_type == 'anndata':

        return adata

    elif return_type == 'multi':

        return adata, image, position

    elif return_type == 'graph':

        if patch_size is None:
            patch_size = set_patch(adata)

        builder = Build_graph(adata, image, position, n_neighbors,
                              patch_size, train_mode)
        return builder.pack()


def read_cross(ref_dir: Optional[str] = None, tgt_dir: Optional[str] = None,
               ref_name: Optional[str] = None, tgt_name: Optional[str] = None,
               ref: Optional[ad.AnnData] = None, tgt: Optional[ad.AnnData] = None, spa_key: str = 'spatial',
               preprocess: bool = True, n_genes: int = 3000, patch_size: Optional[int] = None,
               n_neighbors: int = 4, return_type: Literal['anndata', 'graph'] = 'graph', **kwargs):
    seed_everything(0)

    ref, ref_img, ref_pos = read(ref_dir, ref_name, ref, False, 'multi',
                                 spa_key=spa_key, n_neighbors=n_neighbors)
    tgt, tgt_img, tgt_pos = read(tgt_dir, tgt_name, tgt, False, 'multi',
                                 spa_key=spa_key, n_neighbors=n_neighbors)
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
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                sc.pp.filter_genes(ref, min_cells=10)
                sc.pp.highly_variable_genes(ref, n_top_genes=n_genes, subset=True)
            tgt = tgt[:, ref.var_names]
    
    if return_type == 'anndata':

        return ref, tgt

    elif return_type == 'graph':
    
        if patch_size is None:
            patch_size = set_patch(ref)

        ref_b = Build_graph(ref, ref_img, ref_pos, patch_size=patch_size, **kwargs)
        tgt_b = Build_graph(tgt, tgt_img, tgt_pos, patch_size=patch_size, **kwargs)
        return ref_b.pack(), tgt_b.pack()


def read_multi(input_dir: Optional[str] = None, data_name: Optional[List[str]] = None,
               adata: Optional[List[ad.AnnData]] = None, patch_size: Optional[int] = None,
               preprocess: bool = True, n_genes: int = 3000, n_neighbors: int = 4,
               return_type: Literal['anndata', 'graph'] = 'graph', spa_key: str = 'spatial', **kwargs):
    seed_everything(0)

    # initialize dataname when 'data_name = None'
    if data_name is None:
        if adata is None:
            RuntimeError('Please set the read file/path.')
        else:
            data_name = [None] * len(adata)
    else:
        adata = [None] * len(data_name)

    adatas, images, positions = [], [], []
    for i in range(len(data_name)):
        d, img, pos = read(input_dir, data_name[i], adata[i], False, 'multi',
                           spa_key=spa_key, n_neighbors=n_neighbors)
        adatas.append(d)
        images.append(img)
        positions.append(pos)

    for img in images:
        if img is None:
            images = None
            break

    if preprocess:
        adatas = [preprocess_data(d) for d in adatas]
        ref = adatas[0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            sc.pp.filter_genes(ref, min_cells=10)
            sc.pp.highly_variable_genes(ref, n_top_genes=n_genes, subset=True)
        adatas = [d[:, list(ref.var_names)] for d in adatas]

    if return_type == 'anndata':

        return adatas

    elif return_type == 'graph':

        if patch_size is None:
            patch_size = set_patch(adatas[0])

        builder = Build_multi_graph(adatas, images, positions,
                                    patch_size=patch_size, **kwargs)
        return builder.pack()


