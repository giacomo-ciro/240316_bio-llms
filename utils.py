import random
import numpy as np
import torch

# Dict with attributes
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Given a list of genes, retrieve corresponding TFs
def get_z(g):
    """
    g:
        [B, r]:
                vector of gene tokens (gene_ids)
    """
    # TODO
    # import transcriptional_interactions and filter vocabulary
    # (maybe a cache to avoid importing and checking vocabulary every time?)
    # init (B, r, r) z tensor
    # loop through all genes and fill z

    pass