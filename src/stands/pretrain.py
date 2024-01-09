import os
import dgl
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from typing import List, Optional

from .model import STNet
from ._read import read_multi
from ._utils import seed_everything


def pretrain(input_dir: str, data_name: List[str],
             n_epochs: int = 200,
             patch_size: Optional[int] = None,
             batch_size: int = 128,
             learning_rate: float = 1e-4,
             GPU: bool = True,
             random_state: int = None,
             weight_dir: Optional[str] = None,
             ):
    """
    Pretrain STNet model using spatial data.
    After the completion of pre-training, the weights will be automatically saved.

    Parameters:
        input_dir (str): Directory path for the input spatial data.
        data_name (List[str]): List of names of spatial data.
        n_epochs (int): Number of training epochs.
        patch_size (Optional[int]): Patch size for H&E images.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        GPU (bool): Whether to use GPU for training.
        random_state (int): Random seed for reproducibility.
        weight_dir (Optional[str]): Directory path to save the pretrained model weights.
    """
    if GPU:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("GPU isn't available, and use CPU to train Docs.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    if random_state is not None:
        seed_everything(random_state)
    
    # Initialize dataloader for train data
    train = read_multi(input_dir, data_name, patch_size)
    graph = train['graph']
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataset = dgl.dataloading.DataLoader(
        graph, graph.nodes(), sampler,
        batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=0, device=device)

    net = STNet(train['patch_size'], train['gene_dim']).to(device)
    opt_G = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    G_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer = opt_G, T_max = n_epochs)
    L1 = nn.L1Loss().to(device)

    tqdm.write('Begin to pretrain STANDS...')
    
    with tqdm(total=n_epochs) as t:
        for _ in range(n_epochs):
            t.set_description(f'Pretrain STNet')

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

    # return net.state_dict()
    save_module = ['GeneEncoder', 'GeneDecoder',
                   'ImageEncoder', 'ImageDecoder', 'Fusion']
    if weight_dir is None:
        weight_dir = os.path.dirname(__file__) + '/model.pth'
    net.save_weights(weight_dir, save_module)
    
    tqdm.write(f'The pretrained weights for STANDS have been automatically saved at {weight_dir}!')