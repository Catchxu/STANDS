class SCConfigs(object):
    def __init__(self, gene_dim, out_dim):
        self.gene_dim = gene_dim
        self.out_dim = out_dim




class STConfigs(object):
    def __init__(self, gene_dim, out_dim):
        self.gene_dim = gene_dim
        self.out_dim = out_dim

        self.GATEncoder = {'nheads': [4, 1]}




class FullConfigs(object):
    def __init__(self, gene_dim, out_dim, patch_size, cross_attn):
        self.gene_dim = gene_dim
        self.out_dim = out_dim

        self.GATEncoder = {'nheads': [4, 1]}

        self.patch_size = patch_size
        self.ImageEncoder = {
            'n_ResidualBlock': 8, 
            'n_levels': 2,
            'input_channels': 3, 
            'MultiResSkips': True,
            'GAT_nhead': 4            
        }
        self.ImageDecoder = {
            'n_ResidualBlock': 8, 
            'n_levels': 2,
            'output_channels': 3, 
            'MultiResSkips': True
        }

        self.cross_attn = cross_attn
        self.TFBlock = {
            'num_layers': 3,
            'nheads': 4,
            'hidden_dim': 1024, 
            'dropout': 0.1
        }




class MBConfigs(object):
    def __init__(self):
        self.MBBlock = {
            'mem_dim': 512,
            'shrink_threshold': 0.005,
            'temperature': 0.5
        }
