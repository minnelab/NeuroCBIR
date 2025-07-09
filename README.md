# NeuroCBIR

# Multi-Positive Contrastive Losses

https://github.com/GSidiropoulos/typo-robust-multi-positive-DR

https://github.com/LeiWangR/cl

https://github.com/google-research/syn-rep-learn/blob/main/StableRep/models/losses.py

# Experiments

To do something to deal with the domain adaption, check this paper (Domain-invariant feature learning in brain MR imaging for
content-based image retrieval)

Multiple comparison for evaluation. Focus on the whole brain:

**Testing CBIR precision / success performance for:**

- AE features
- U-Map proj features from AE feautes
- Features from CL with MultiPosConLoss
- Features from CL with MultiPosConLoss + DANN
- Features from CL with MultiPosConLoss + Age
- Features from CL with MultiPosConLoss + Disease
- Features from CL with MultiPosConLoss + Age + Disease

**Testing domain invariance**

To confirm that z is domain-invariant after training:

- Train a domain classifier (without GRL) on frozen z embeddings.

- If accuracy is close to random (e.g. 50% for 2 domains), the DANN worked.

- Also visualize with t-SNE or UMAP and color by domain — domain clusters should disappear.

