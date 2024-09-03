from typing import Optional

import torch 
import torch.nn as nn

class OuterProductMean(nn.Module):
    """
    Input:
        RNA:
            [*, N_res, c_m] tensor of sequence representation
    
    Output:
        Pair-wise updates:
            [*, N_res, N_res, c_z] biases to update pair-wise representation

    Pseudocode:
        Take two columns i,j of RNA
        project to c_hidden dimension
        Outer-product(i,j) --> matrix (c, c)
        Flatten --> vector (c**2, )
        nn.Linear(c**2, c) --> vector (c, )
        Pair-wise updates [i,j] = above
    """
    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension (for the subsequent pair-wise update)
            c_hidden:
                Hidden channel dimension
        """
        super(OuterProductMean, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, c_hidden)
        self.linear_2 = nn.Linear(c_m, c_hidden)
        self.linear_out = nn.Linear(c_hidden ** 2, c_z)

    def _opm(self, a, b):
        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...ab,...cd->...acbd", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,)) # flatten last two dim

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        return outer

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_res, C_m] MSA embedding

        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """

        # [*, N_res, C_m]
        ln = self.layer_norm(m)

        a = self.linear_1(ln)         
        b = self.linear_2(ln) 

        del ln

        outer = self._opm(a, b)

        return outer


class RowAttentionWithPairBias:
    """
    Input:
        RNA:
            (r, c)
        
        TF:
            (r, r, c)
    
    Output:
        Attention:
            (r, c)

    Compute attn between any two vectors i,j in the input RNA using TF[i,j] as pair bias in the attn computation
    """
    def __init__(self) -> None:
        pass

