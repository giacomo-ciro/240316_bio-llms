from typing import Optional
import math

import torch 
import torch.nn as nn

# B = batch_size
# r = number of residues in the sequence
# c = input embedding dimensions

class OuterProductMean(nn.Module):
    """
    
    Compute outer product mean of a given tensor.
    
    """
    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m:
                Input embedding channel dimension
            c_z:
                Pair embedding channel dimension
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
        # [*, r, r, c, c]
        outer = torch.einsum("...ab,...cd->...acbd", a, b)

        # [*, r, r, c * c]
        outer = outer.reshape(outer.shape[:-2] + (-1,)) # flatten last two dim

        # [*, r, r, c_z]
        outer = self.linear_out(outer)

        return outer

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m:
                [B, r, c] Input embedding

        Returns:
            [B, r, r, c_z] pair embedding update
        """

        # [*, r, c_m]
        ln = self.layer_norm(m)

        a = self.linear_1(ln)         
        b = self.linear_2(ln) 

        del ln

        outer = self._opm(a, b)

        return outer


class RowAttentionWithPairBias(nn.Module):
    """

    Compute a given sequence self-attention using provided biases.
    
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

        # Initial Normalization
        self.layer_norm_m = nn.LayerNorm(self.c_in)

        # Bias
        self.layer_norm_z = nn.LayerNorm(self.c_z) if self.pair_bias else None
        self.linear_z = nn.Linear(self.c_z, self.no_heads, bias=False) if self.pair_bias else None

        # Queries, Keys, Values
        self.linear_q = nn.Linear(self.c_in, self.c_hidden * self.no_heads, bias=False)
        self.linear_k = nn.Linear(self.c_in, self.c_hidden * self.no_heads, bias=False)
        self.linear_v = nn.Linear(self.c_in, self.c_hidden * self.no_heads, bias=False)
        
        # Gating
        self.linear_g = nn.Linear(self.c_in, self.c_hidden * self.no_heads) if self.gating else None
        
        # Final projection
        self.linear_o = nn.Linear(self.c_hidden * self.no_heads, self.c_in)

    def forward(self,
            m: torch.Tensor,
            z: Optional[torch.Tensor] = None,
    )->torch.Tensor:
        """
        Args:
            m:
                [B, r, c] Input embedding
            z:
                [B, r, r, c_z] pair embedding. Required only if
                pair_bias is True
        """
        if z is None and self.pair_bias is True:
            raise Warning("z required when pair bias is true")
        
        # [B, r, c]
        m = self.layer_norm_m(m)

        # [B, r, c_hid * H]
        q = self.linear_q(m)
        k = self.linear_k(m)
        v = self.linear_v(m)

        # [*, r, H, c_hid], where H = no_heads
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, r, c_hid]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        q /= math.sqrt(self.c_hidden)   # scaled attn

        # [*, H, c_hid, r]
        k = torch.permute(k, (0, 1, 3, 2))  # flip last two dims for matmul

        # [*, H, r, r]
        a = torch.matmul(q, k)

        if self.pair_bias:
            # [B, r, r, c_z]
            z = self.layer_norm_z(z)
            # [B, r, r, H]
            z = self.linear_z(z)
            # [B, H, r, r]
            z = torch.permute(z, (0, 3, 1, 2))
            # [B, H, r, r]
            a += z

        # [B, H, r, r]
        a = nn.functional.softmax(a, -1)

        # [B, H, r, c_hid]
        a = torch.matmul(a, v)

        # [B, r, H, c_hid]
        a = a.transpose(-2, -3)
        
        if self.gating:
            # [B, r, c_hid * H]
            g = nn.functional.sigmoid(self.linear_g(m))
            # [B, r, H, c_hid]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            # [B, r, H, c_hid]
            a = a * g

        # [B, r, H * c_hid]
        a = a.reshape(a.shape[:-2] + (-1,))     # flatten H dim

        # [B, r, c_m]
        a = self.linear_o(a)
        
        return a


class Transition(nn.Module):
    """
    Feed-forward network.
    """
    def __init__(self, c_m, n=4):
        """
        Args:
            c_m:
                Input channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension (default is 4).
        """
        super(Transition, self).__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = nn.LayerNorm(self.c_m)
        self.linear_1 = nn.Linear(self.c_m, self.n * self.c_m)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.n * self.c_m, self.c_m)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m:
                [B, r, r, c] Input 

        Returns:
            m:
                [B, r, r, c] Update
        """
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m)

        return m
    

class BioFormerBlock(nn.Module):
    """
    
    A single BioFormer block.
    It performs gated-self attention with pair bias, transition and outer-product mean sequentially via residual connections.

    """
    def __init__(self, c_m, c_z, c_hidden, no_heads):
        """
        Args:
            c_m:
                Input channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
        """
        super(BioFormerBlock, self).__init__()

        self.opm = OuterProductMean(
                                c_m=c_m,
                                c_z=c_z,
                                c_hidden=c_hidden
                                )
        self.attn = RowAttentionWithPairBias(
                                        c_in=c_m,
                                        c_hidden=c_hidden,
                                        no_heads=no_heads,
                                        pair_bias=True,
                                        c_z=c_z, 
                                        gating=True
                                        )
        self.trans = Transition(
                            c_m=c_m,
                            n=4
                            )

    def forward(self,m, z):
        """
        Args:
            m:
                [B, r, c_m] RNA-seq input
            z:
                [B, r, r, c_z] pair representation
        Returns:
            m:
                [B, r, c_m] updated RNA-seqc
            z:
                [B, r, r, c_z] updated pair representation
        """
        m += self.attn(m, z)
        m += self.trans(m)
        z += self.opm(m)

        return m, z
        
      
class BioFormerStack(nn.Module):
    """
    
    A series of BioFormer blocks.

    """
    def __init__(self,
            c_m,
            c_z,
            c_hidden,
            no_heads,
            no_blocks
            ):
        super(BioFormerStack, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = BioFormerBlock(c_m=c_m, c_z=c_z, c_hidden=c_hidden, no_heads=no_heads)
            self.blocks.append(block)

    def forward(self, m, z):

        for b in self.blocks:
            m, z = b(m, z)
        
        return m, z