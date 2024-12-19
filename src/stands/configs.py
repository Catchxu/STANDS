class SCConfigs(object):
    def __init__(self, gene_dim):
        self.gene_dim = gene_dim
        self.out_dim = [512, 256]
        self.z_dim = self.out_dim[-1]


class STConfigs(object):
    def __init__(self, gene_dim):
        self.gene_dim = gene_dim
        self.out_dim = [512, 256]
        self.z_dim = self.out_dim[-1]

        self.GATEncoder = {'nheads': [4, 1]}


class FullConfigs(object):
    def __init__(self, gene_dim, patch_size):
        self.gene_dim = gene_dim
        self.out_dim = [512, 256]
        self.z_dim = self.out_dim[-1] * 2

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

        self.cross_attn = True
        self.TFBlock = {
            'num_layers': 3,
            'nheads': 4,
            'hidden_dim': 512,
            'dropout': 0.3
        }


class DisSCConfigs(object):
    def __init__(self, gene_dim):
        self.gene_dim = gene_dim
        self.out_dim = [512, 256]
        self.z_dim = self.out_dim[-1]


class DisFullConfigs(object):
    def __init__(self, gene_dim, patch_size):
        self.gene_dim = gene_dim
        self.out_dim = [512, 256]
        self.z_dim = self.out_dim[-1] * 2

        self.patch_size = patch_size
        self.ImageEncoder = {
            'n_ResidualBlock': 8, 
            'n_levels': 2,
            'input_channels': 3, 
            'MultiResSkips': True,
            'GAT_nhead': 4
        }


class MBConfigs(object):
    def __init__(self, z_dim):
        self.MBBlock = {
            'z_dim': z_dim,
            'mem_dim': 512,
            'shrink_threshold': 0.05,
            'temperature': 0.07
        }


class DisConfigs(object):
    def __init__(self, z_dim, only_ST=False, only_SC=False):
        self.in_dim = z_dim
        self.out_dim = [256, 256, 256, 16]
        self.dim_list = [self.in_dim] + self.out_dim


class GMMConfigs(object):
    def __init__(self):
        self.GMM = {
            'max_iter': 100,
            'tol': 1e-3,
            'prior_beta': [1,10]
        }


class ClusterConfigs(object):
    def __init__(self, z_dim):
        self.alpha = 1
        self.KMeans_n_init = 20
        self.cross_attn = True
        self.TFBlock = {
            'g_dim': z_dim,
            'p_dim': z_dim,
            'num_layers': 3,
            'nheads': 8,
            'hidden_dim': 512, 
            'dropout': 0.1
        } 

        # training process
        self.learning_rate = 1e-4
        self.n_epochs = 3000
        self.update_interval = 10
        self.weight_decay = 1e-4