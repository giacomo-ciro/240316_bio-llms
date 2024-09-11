# 🧬 Bio-LLMs
Exploring Large Language Models for Gene Expression prediction.  
In particular, I am trying to exploit insights and ideas behind Alpha Fold 2 to effectively incorporate prior biological knowledge about Transcription Factors (TF) activity into a Transformer Model to predict Masked RNA-seq values.

## Model
I've implemented two models:
- `model.py` is the default scGPT with some changes (e.g. no custom attention);
- `model_bioformer.py` is my proposed idea inspired by AlfaFold 2's EvoFormer. It takes the same input as scGPT, i.e. a (B, r, c) tensor `m` which summarizes RNA-seq and Gene Tokens, but also an additional pair representation of each cell RNA-seq, i.e. a (B, r, r, c) tensor `z`. Then, it processes both input simultaneously allowing them to influence each other. In particular, `m` goes through a self-attention layer where `z` is used as bias term before softmax. After this, `m` goes through a classic ffnn. Finally, an outer product mean is computed between all columns of `m` to obtain an update for `z`. This block is repeated N times. Notice how `z` influences `m` via the pair bias in the attention, while `m` influences `z` via the outer product mean update.   

NOTE 1: the outer product between two c-dim vectors is a (c,c) matrix, which is flattened and projected back to c-dim. Hence, a c**2 linear layer is required, which implies exponential growth of the number of parameters as c increase.  

NOTE 2: bioformer is composed by 3 modules, self-attn with pair bias, ffnn, opm. Hence, it is a simple transformer layer except for the bias term and the opm.

NOTE 3: the key strength of bioformer is the baility to reason over the pair representation, hence it is key to have a high enough N (bioformer blocks in the stack) to allow this.

## Tests
To test performance I conduct the following tests:  
1. **same no. params** --> I force scGPT and BioFormer to have the same number of parameters. Given what observed above, c_bioformer < c_scgpt. After 5 epochs of training on HYPOXIA_9K, scGPT has lower MSE, yet BioFormer has lower MRE. The models are somehow of comparable performance. Given NOTE 2 and without considering omp, bioformer is a transformer with less parameters than scGPT, yet of comparable performance. I conclude the opm layer and the bias term effectively mitigate the lower number of attn and ffnn parameters, suggesting this type of layers might be useful in the task at hand.  
2. **same no. of attn params** --> I force scGPT and BioFormer to have the same number of parameters when not considering the opm layer. Hence, BioFormer is allowed to have more parameters is the difference is due to the opm layer only. In this scenario, BioFormer performs better than scGPT, they reach same mse, bioformer lower mre, and in both cases is faster to converge. Hence, same self-attn, same ffnn, opm looks like is actually useful!
3. **ablation test** --> bioformer without opm and bias, is it exactly like scGPT?
bioformer without opm and bias perform the same as bioformer with opm and bias, but for the MRE which is slightly higher. Moreover, withouth opm and bias converges faster. Maybe the key is the gating? no gating does not solve the problem. Maybe the factor of the ffnn, I use n=4 but in scGPT it's n=1. Testing bioformer with n=1: it's not the key factor. still bioformer converges way faster
4. **increasing N** --> in all these tests I stuck with a low N=2 for both scgpt and bioformer. I tried to increase N=8 and bioformer 32 is comparable to scgpt 256 with N=2. I have to further test for example expanding both N's.

---
**Notation**
- B = batch size;
- r = sequence length;
- c = embedding dimension.
**Info**
In the original scGPT paper, they implement the ffnn in the transformer encoder with the d_hid = d_model (not 4\*d_model as in the original transformer paper). In this implementation of scGPT I stick with the original d_hid = 4\*d_model for both scGPT and BioFormer.