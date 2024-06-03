import copy
import dgl
import torch
from typing import Optional, Union, Dict, Any

from .model import GeneratorAD, Cluster
from ._utils import select_device, seed_everything


class Subtype:
    def __init__(self, 
                 generator: GeneratorAD, 
                 n_subtypes: int = 2,
                 GPU: Union[bool, str] = True, 
                 random_state: Optional[int] = None,
            ):

        self.n_subtype = n_subtypes
        self.device = select_device(GPU)
        self.G = copy.deepcopy(generator).to(self.device)
        self.C = Cluster(self.G, self.n_subtype)

        if random_state is not None:
            seed_everything(random_state)
        self.seed = random_state

    def fit(self, data: Dict[str, Any]):
        '''Detect subtypes of samples'''

        graph = data['graph']

        self.G.eval()
        self.C.train()
        z, res_z = self.generate_z_res(graph)
        q = self.C.fit(z, res_z)
        return q

    @torch.no_grad()
    def generate_z_res(self, graph: dgl.DGLGraph):
        '''Generate reconstructed data'''
        self.G.eval()
        if self.G.extract.only_ST:
            z, fake_g = self.G.STforward(graph, graph.ndata['gene'])
            res_g = graph.ndata['gene'] - fake_g.detach()
            res_z = self.C.STforward(graph, res_g)

        else:
            z, fake_g, fake_p = self.G.Fullforward(
                graph, graph.ndata['gene'], graph.ndata['patch']
            )
            res_g = graph.ndata['gene'] - fake_g.detach()
            res_p = graph.ndata['patch'] - fake_p.detach()
            res_z = self.C.Fullforward(graph, res_g, res_p)

        return z, res_z
