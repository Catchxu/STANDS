'''
STANDS is an innovative framework seamlessly integrated detecting, aligning and subtyping anomlous tissue domains.
'''

from .main import ADNet, AlignNet, BCNet
from .pretrain import pretrain
from ._read import read, read_cross, read_multi
from .evaluate import evaluate


__all__ = ['ADNet', 'BCNet',
           'read', 'read_cross', 'read_multi',
           'pretrain', 'evaluate']