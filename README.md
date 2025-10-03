# Description

- Implements a Vision Transformer (ViT) from scratch.
- Trains and evaluates the model on CIFAR-10.

- The code (q1.ipynb) is divided in two parts indicated by the headings : 'Training part', 'Run the code'. For explanation please see the following lines (and for how to run the code, i.e, just for testing, please jump to the 'Run the code' section of this readme):

# Training part 
## Data / Model

img_size = 32
CIFAR-10 images are 32×32, so you process them at native resolution.

num_classes = 10
Matches CIFAR-10’s 10 categories.

patch_size = 4
Splits each 32×32 image into non-overlapping 4×4 patches → (32/4)² = 8×8 = 64 tokens.
More tokens = finer spatial detail than 8×8 patches would give, but a bit more compute.

embed_dim = 384
Channel width of token embeddings and transformer hidden states. A solid “small ViT” width for CIFAR-10.

depth = 8
Number of transformer blocks (encoder layers). Moderate depth keeps training stable and fast.

num_heads = 6
Multi-head attention splits 384 dims across 6 heads → 64 dims per head (384/6). A sweet spot for efficiency.

mlp_ratio = 4.0
MLP hidden size = 4×embed_dim = 1536. Standard ViT setting balancing capacity vs. compute.

drop_rate = 0.1
Plain dropout inside blocks (often on MLP activations). Helps regularize small datasets like CIFAR-10.

attn_drop_rate = 0.0
Dropout on attention weights. Kept off here; attention dropout often unnecessary with other regularizers.

drop_path_rate = 0.1
Stochastic depth across layers. Typically ramped from 0 → 0.1 with layer index; helps deeper models generalize.

## Optimization

batch_size = 128
Common batch size that fits on T4/Colab. If you raise it, you can often scale LR up linearly.

epochs = 200
Enough for ViT on CIFAR-10 to converge with strong augments.

warmup_epochs = 10
Gradually ramps LR from near-zero to base_lr to avoid early instability (crucial for transformers).

base_lr = 5e-4
Peak learning rate after warmup (with AdamW). Reasonable for embed_dim=384, bs=128.

min_lr = 5e-6
Floor LR for cosine decay (often you’ll use cosine: warmup → cosine down to this).

weight_decay = 0.05
AdamW’s decoupled L2. ViT papers often use 0.05–0.1; 0.05 is conservative and stable.

betas = (0.9, 0.999)
Adam/AdamW momentum coefficients. Standard defaults; work well.

## Regularization

label_smoothing = 0.1
Softens one-hot targets, reduces overconfidence, improves calibration and sometimes top-1 on small datasets.

## Augmentations

use_randaugment = True
Strong, simple augmentation policy—key for getting ViTs to behave on small datasets like CIFAR-10.

randaugment_n = 2
Number of ops per image. 2 is a good baseline.

randaugment_m = 9
Magnitude (strength) of ops, typically 0–10. 9 is fairly strong but still reasonable for CIFAR-10.
