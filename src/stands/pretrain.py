import os
import dgl
import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from typing import List, Optional, Union

from .model import Extractor
from .configs import FullConfigs
from ._read import read_multi
from ._utils import seed_everything


def pretrain(adata_list: List[ad.AnnData],
             n_epochs: int = 50,
             patch_size: Optional[int] = None,
             batch_size: int = 128,
             learning_rate: float = 1e-4,
             GPU: Union[bool, str] = True,
             random_state: int = None,
             weight_dir: Optional[str] = None
             ):
    """
    Pretrain STANDS basic extractors using spatial data.
    After the completion of pre-training, the weights will be automatically saved.

    Parameters:
        adata_list (List[ad.AnnData]): input spatial data (to be trained).
        n_epochs (int): Number of training epochs.
        patch_size (Optional[int]): Patch size for H&E images.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        GPU (Union[bool, str]): Whether to use GPU for training, and GPU ID (i.e., cuda:0)
        random_state (int): Random seed for reproducibility.
        weight_dir (Optional[str]): Directory path to save the pretrained model weights.
    """
    if GPU:
        if torch.cuda.is_available():
            if isinstance(GPU, str):
                device = torch.device(GPU)
            else:
                device = torch.device('cuda:0')
        else:
            print("GPU isn't available, and use CPU to train Docs.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    if random_state is not None:
        seed_everything(random_state)

    # Initialize dataloader for train data
    train = read_multi(adata_list, patch_size, preprocess=False)
    graph = train['graph']
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataset = dgl.dataloading.DataLoader(
        graph, graph.nodes(), sampler,
        batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=0, device=device)

    configs = FullConfigs(train['gene_dim'], train['patch_size'])
    net = Extractor(configs).to(device)
    opt_G = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    G_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=opt_G, T_max=n_epochs)
    L1 = nn.L1Loss().to(device)

    tqdm.write('Begin to pretrain STANDS...')

    with tqdm(total=n_epochs) as t:
        for _ in range(n_epochs):
            t.set_description(f'Pretrain STANDS')

            for _, _, blocks in dataset:
                opt_G.zero_grad()

                real_g = blocks[0].srcdata['gene']
                real_p = blocks[1].srcdata['patch']

                fake_g, fake_p = net.pretrain(blocks, real_g, real_p)
                Loss = (L1(blocks[1].dstdata['gene'], fake_g) + \
                        L1(blocks[1].dstdata['patch'], fake_p))

                Loss.backward()
                opt_G.step()

            G_scheduler.step()
            t.set_postfix(Loss = Loss.item())
            t.update(1)

    if weight_dir is None:
        weight_dir = os.path.dirname(__file__) + '/model/backbone/model.pth'
    torch.save(net.state_dict(), weight_dir)
    tqdm.write(f'The pretrained weights for STANDS have been automatically saved at {weight_dir}!')