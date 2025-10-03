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
Channel width of token embeddings and transformer hidden states. 

depth = 8
Number of transformer blocks (encoder layers). Moderate depth keeps training stable and fast.

num_heads = 6
Multi-head attention splits 384 dims across 6 heads → 64 dims per head (384/6). 

mlp_ratio = 4.0
MLP hidden size = 4×embed_dim = 1536. Standard ViT setting balancing 

drop_rate = 0.1
Plain dropout inside blocks (often on MLP activations). 

attn_drop_rate = 0.0
Dropout on attention weights. Kept off here; attention dropout often unnecessary with other regularizers.

drop_path_rate = 0.1
Stochastic depth across layers. Typically ramped from 0 → 0.1 with layer index; helps deeper models generalize.

### Optimization

batch_size = 128
Common batch size that fits on T4/Colab.

epochs = 200
Enough for ViT on CIFAR-10 to converge with strong augments.

warmup_epochs = 10
Gradually ramps LR from near-zero to base_lr to avoid early instability.

base_lr = 5e-4
Peak learning rate after warmup (with AdamW). Reasonable for embed_dim=384, bs=128.

min_lr = 5e-6
Floor LR for cosine decay (often you’ll use cosine: warmup -> cosine down to this).

weight_decay = 0.05
AdamW’s decoupled L2. ViT papers often use 0.05–0.1; 0.05 is conservative and stable.

betas = (0.9, 0.999)
Adam/AdamW momentum coefficients. Standard defaults; work well.

### Regularization

label_smoothing = 0.1
Softens one-hot targets, reduces overconfidence, improves calibration and sometimes top-1 on small datasets.

### Augmentations

use_randaugment = True
Strong, simple augmentation policy—key for getting ViTs to behave on small datasets like CIFAR-10.

randaugment_n = 2
Number of ops per image. 2 is a good baseline.

randaugment_m = 9
Magnitude (strength) of ops, typically 0–10. 9 is fairly strong but still reasonable for CIFAR-10.


## Data Preparation

The CIFAR-10 dataset is loaded and preprocessed before training. Each image is normalized using the dataset’s channel-wise mean and standard deviation:

**Mean**: (0.4914, 0.4822, 0.4465)
**Standard deviation**: (0.2470, 0.2435, 0.2616)

For the training set, these data augmentation techniques are applied to improve model generalization:

**RandomCrop** with padding of 4 pixels simulates small translations.
**RandomHorizontalFlip** randomly flips images horizontally with a 50% probability.
If **use_randaugment** is enabled in the configuration, RandAugment applies additional random transformations to further increase data diversity.

Images are then converted to tensors and normalized.

For the test set, only tensor conversion and normalization are applied to ensure consistent evaluation without random augmentations.

Finally, DataLoader objects are created for both training and test sets, handling batching, shuffling (for training), and parallel data loading. The dataset consists of 50,000 training images and 10,000 test images.


## Vision Transformer Architecture

The model is implemented fully from scratch, following the original Vision Transformer (ViT) design. The core components are as follows:

**Patch Embedding (PatchEmbed)** – Splits the input image into non-overlapping patches and projects each patch into a vector embedding using a convolution layer. This converts a 2D image into a sequence of patch tokens suitable for a transformer.

**Multi-Layer Perceptron (MLP)** – A simple two-layer feed-forward network with GELU activation and dropout, used inside each transformer block after the attention layer.

**Multi-Head Self-Attention (MultiHeadSelfAttention)** – Computes attention across all patch tokens. The input is projected into queries, keys, and values, split across multiple heads, and combined to learn global relationships among patches.

**Stochastic Depth (DropPath)** – Implements layer-level dropout, randomly dropping entire residual branches during training to improve generalization and stabilize deeper models.

**Transformer Block (Block)** – A standard transformer encoder block consisting of:

Layer normalization
Multi-head self-attention
DropPath regularization
Feed-forward MLP with residual connection

Vision Transformer (VisionTransformer) – Combines all components into the final model:

A learnable [CLS] token is prepended to the sequence and used for classification.

Positional embeddings are added to retain spatial information.

A stack of transformer encoder blocks processes the token sequence.

The final embedding of the CLS token is normalized and passed through a linear classification head.


## Training Utilities

These utility components are used to support the training of the Vision Transformer:

**LabelSmoothingCE**
A custom cross-entropy loss function with label smoothing. Instead of assigning a probability of 1.0 to the correct class, a small portion (smoothing, e.g., 0.1) is distributed across all other classes.
This regularization technique helps prevent the model from becoming over-confident and often improves generalization, especially on smaller datasets like CIFAR-10.

**evaluate()**
A utility function to evaluate the model on a validation or test dataset without computing gradients. It switches the model to evaluation mode, iterates over the data loader, computes predictions, and returns the classification accuracy.

**WarmupCosine**
A learning rate scheduler that combines a linear warm-up phase with a cosine decay schedule:

During the warm-up phase (first warmup_epochs), the learning rate increases linearly from 0 to the base learning rate.
After warm-up, the learning rate follows a cosine decay curve down to min_lr by the end of training.

## Training

The Vision Transformer model is trained on the CIFAR-10 dataset for 200 epochs using the AdamW optimizer with a base learning rate of 5e-4, cosine learning rate scheduling with linear warmup for the first 10 epochs, and a weight decay of 0.05.
The loss function used is label-smoothed cross-entropy (smoothing = 0.1), and automatic mixed precision (AMP) is enabled to speed up training and reduce memory usage.
The training loop computes the loss and updates model weights at each step. After every epoch, the model is evaluated on the test set, and the best-performing checkpoint is saved. ('vit_cifar10_best.pt' <- use this for loading the state dict in eval mode later).


