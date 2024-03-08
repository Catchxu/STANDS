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
from .main import SubNet


def pretrain(input_dir: Optional[str] = None, 
             data_name: Optional[List[str]] = None,
             train: Optional[dict] = None,
             n_epochs: int = 200,
             patch_size: Optional[int] = None,
             batch_size: int = 128,
             learning_rate: float = 1e-4,
             GPU: bool = True,
             random_state: int = None,
             weight_dir: Optional[str] = None,
             ):
    """
    Pretrain STANDS basic networks using spatial data.
    After the completion of pre-training, the weights will be automatically saved.

    Parameters:
        input_dir (Optional[str]): Directory path for the input spatial data.
        data_name (Optional[List[str]]): List of names of spatial data.
        train (Optional[dict]): input spatial data (to be trained).
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
    if train is None:
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

    if weight_dir is None:
        weight_dir = os.path.dirname(__file__) + '/model.pth'
    torch.save(net.state_dict(), weight_dir)
    tqdm.write(f'The pretrained weights for STANDS have been automatically saved at {weight_dir}!')




def pretrain_cluster(input_dir: str,
                     data_name: List[str],
                     type_key: str,
                     generator: nn.Module,
                     n_epochs: int = 200,
                     patch_size: Optional[int] = None,
                     learning_rate: float = 1e-4,
                     GPU: bool = True,
                     random_state: int = None,
                     weight_dir: Optional[str] = None,
             ):
    # Initialize dataloader for train data
    train = read_multi(input_dir, data_name, patch_size)
    n_subtypes = len(train['adata'].obs[type_key].unique())

    parameters = {
        'generator': generator,
        'n_subtypes': n_subtypes,
        'learning_rate': learning_rate,
        'GPU': GPU,
        'random_state': random_state
    }
    model = SubNet(**parameters)
    ClusterNet = model.pretrain(train, type_key, n_epochs)
    
    # Save both G and Fusion weights in a single file
    save_state = {
        'generator': ClusterNet.C.G.state_dict(),
        'fusion': ClusterNet.C.Fusion.state_dict(),
    }
    if weight_dir is None:
        weight_dir = os.path.dirname(__file__) + '/cluster.pth'

    torch.save(save_state, weight_dir)