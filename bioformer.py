from typing import Optional
import math

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


class RowAttentionWithPairBias(nn.Module):
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
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        pair_bias=False,
        c_z=None,
        gating=True,
    )-> None:
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            gating:
                Whether to use gated attention or not
        """
        super(RowAttentionWithPairBias, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.gating = gating

        self.layer_norm_m = nn.LayerNorm(self.c_in)

        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = nn.LayerNorm(self.c_z)
            self.linear_z = nn.Linear(self.c_z, self.no_heads, bias=False)

        self.linear_q = nn.Linear(self.c_q, self.c_hidden * self.no_heads, bias=False)
        self.linear_k = nn.Linear(self.c_k, self.c_hidden * self.no_heads, bias=False)
        self.linear_v = nn.Linear(self.c_v, self.c_hidden * self.no_heads, bias=False)
        self.linear_o = nn.Linear(self.c_hidden * self.no_heads, self.c_q)

        self.linear_g = None
        if self.gating:
            self.linear_g = nn.Linear(self.c_q, self.c_hidden * self.no_heads, init="gating")

        self.sigmoid = nn.Sigmoid()

    def forward(self,
            m: torch.Tensor,
            z: Optional[torch.Tensor] = None,
    )->torch.Tensor:
        """
        Args:
            m:
                [*, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding. Required only if
                pair_bias is True
        """
        m = self.layer_norm_m(m)

        q = self.linear_q(m)
        k = self.linear_k(m)
        v = self.linear_v(m)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # TODO: check dimensions here, is this line needed?
        # [*, H, Q/K, C_hidden]
        # q = q.transpose(-2, -3)
        # k = k.transpose(-2, -3)
        # v = v.transpose(-2, -3)

        q /= math.sqrt(self.c_hidden)

        # TODO: check dimensions here, is this line needed?
        # [*, H, C_hidden, K]
        # key = permute_final_dims(k, (1, 0))

        # [*, H, Q, K]
        a = torch.matmul(q, k)

        if self.pair_bias:
            z = self.layer_norm_z(z)
            z = self.linear_z(z)
            # TODO: sum to attn score

        a = nn.Functional.softmax(a, -1)

        # [*, H, Q, C_hidden]
        a = torch.matmul(a, v)
        
        if self.gating:
            g = self.sigmoid(self.linear_g(m))
        
            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = o.reshape(o.shape[:-2] + (-1,))     # flatten H dim

        # [*, Q, C_q]
        o = self.linear_o(o)
        return a
