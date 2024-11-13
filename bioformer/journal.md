# Research Journal
Some notes, thoughts and ideas stemming from my research activity at the Buffa Lab at Bocconi University.

# TODO
- cross-validation for n_bins, layer-norm ordering
- include_zero_genes = True
- test set outside of epochs (train, val, test)
- expand dataset (breast cell atlas and ask Federico perturb data he used for the thesis norman and adamson)
- use omnipath references to add weights and updates on the z (later on)
- tweak also params d_z and c_opm

## Memory Complexity Analysis
Currenly, I am getting a `CUDA out-of-memory: trying to allocate 200GiB` when on the cluster using 2xNVIDIA A100 80GB an instance of bioformer with configuration:

- B = 32
- d_model = 128
- n_hvg = 500
- r = n_hvg + 1  

What is taking so much space?? Outer Product Mean and Attention!

### Outer Prouct Mean (np.einsum)
The outer product computation and in particular when it's parallelized using `np.einsum`! Because I am storing $B$ matrices of shape $(r, r)$ whose entries are all $(c, c)$ matrices themselves. Huge matrices of huge matrices are very heavy to store in memory.   
The outuput of this command has shape `[B, r, r, c, c]`. Assuming each element is `float32` and thus takes 4 bytes (since I am using `automatic mixed precision` this is actually an upper bound), this requires:
$$
B \cdot r^2 \cdot c^2 \cdot 4 \text{  \,bytes}
$$
And in particular for my instance:
$$
32 \cdot 501^2 \cdot 128^2 \cdot 4 \text{ \,bytes} \sim
500 \text{ \,GB}
$$

Even when scaling down to:
- B = 16
- d_model = 128
- n_hvg = 400  

I still get the error because:

$$
16 \cdot 400^2 \cdot 128^2 \cdot 4 \text{ \,bytes} \sim
170 \text{ \,GB}
$$

In this case, the error is `CUDA out-of-memory: trying to allocate 21GiB`.  I want to try to run it with 

$$
16 \cdot 500^2 \cdot 64^2 \cdot 4 \text{ \,bytes} \sim
65 \text{ \,GB}
$$
and this should run on one single A100.

### Attention
We can split into 3 main steps generating intermediate tensors:
1. Input of shape $(B, r, c)$ is projected into K, Q, V spaces of dimension $c' = \frac{c}{n\_heads}$ by construction hence generating three tensors of shape $(B, n\_heads, r, c')$
2. Attention scores matrices are computed for each head, generating a $(B,no\_heads,r,r)$ tensor. 
3. Values are weighted using attention scores, hence just scaling the original tensor in the V space.
According to the input we can summarize as follows the memory requirements:

| Batch Size | Input Length | Model Dim | N. Heads | Head Dim |      Q, K, V      | Attention Scores | parallelized outer prod |
|:----------:|:------------:|:---------:|:--------:|----------|:-----------------:|------------------|-------------------------|
|      B     | r            |     c     |     n    | $c' = \frac{c}{n}$ | $B \cdot n\cdot r \cdot c' \cdot 3$ | $B \cdot n \cdot r \cdot r$ | $B \cdot r^2 \cdot c^2$|
|     32     |    500 | 128 |  4   |    32     | 8 MB | 128 MB | 524,288 MB |
|     16     |    400 | 128 |  4   |    32     | 3 MB | 41 MB | 167,772 MB |
|     16     |    500 | 64 |  4   |    32     | 2 MB | 64 MB | 65,536 MB |  

In conclusion, the upper bound to memory requirements is the parallelized outer product. Should I try a non-parallelized implementation? Perhaps, generate the outer product and immediately project it without first computing everything so instead of having a $(c,c)$ I have the flattened and projected single entry.

## AF2 Configuration
In the original AlphaFold 2 paper, the authors train the model with the following parameters configuration:
- N = 48,
- r = 256, 
- c_model = 256,
- c_z = 128
- c_opm = 32
- c_head = 32
- n_head = 8
- batch_size = 128  

They parallelize training by using 128 Google V3 TPUs (16Gib), one per each batch.