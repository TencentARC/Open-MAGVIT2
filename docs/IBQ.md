<div align="center"> 
<h2>IBQ: Taming Scalable Visual Tokenizer for Autoregressive Image Generation</h2>
</div>

<div align="center">

<!-- > [**Taming Scalable Visual Tokenizer for Autoregressive Image Generation**](https://arxiv.org/abs/2412.02692)<br> -->
> [Fengyuan Shi*](https://shifengyuan1999.github.io/), [Zhuoyan Luo*](https://robertluo1.github.io/), [Yixiao Ge](https://geyixiao.com/), [Yujiu Yang](https://sites.google.com/view/iigroup-thu/people), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en), [Limin Wang](https://wanglimin.github.io/)
> <br> Nanjing University, Tsinghua University, ARC Lab Tencent PCG<br>
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2412.02692-b31b1b.svg)](https://arxiv.org/abs/2412.02692)&nbsp;

</div>

This is the official repository for Index Backpropagation Quantization (IBQ), a novel vector quantization (VQ) method that revolutionizes the scalability and performance of visual tokenizers.
<p align="center">
<img src="../assets/IBQ-teaser.png" width=75%>
</p>

## **Highlights**
- 🚀 **Scalable Visual Tokenizers**: IBQ enables scalable training of visual tokenizers, and achieves a large-scale codebook of size (262144) and high-dimensional embeddings (256), ensuring high utilization.
- 💡 **Innovative Approach**: Unlike conventional VQ methods prone to codebook collapse due to the partial-updating, IBQ leverages a straight-through estimator on the categorical distribution, enabling the joint optimization of all codebook embeddings and the visual encoder, for consistent latent space.  
- 🏆 **Superior Performance**: Demonstrates competitive results on ImageNet:  
  - **Reconstruction**: 1.00 rFID, outperforming Open-MAGVIT2 (1.17 rFID)
  - **Autoregressive Visual Generation**: 2.05 gFID, outperforming previous vanilla autoregressive transformers.

<p align="center">
<img src="../assets/IBQ-gradient-flow.png" width=75%>
</p>

This repository provides the scripts and checkpoints to replicate our results.

## 🔥 Quick Start
<!-- * `Stage I Tokenizer Training`: -->
### Stage I: Training of Visual Tokenizer

#### 🚀 Training Scripts
* $256\times 256$ Tokenizer Training
```
bash scripts/train_tokenizer/IBQ/run_16384.sh MASTER_ADDR MASTER_PORT NODE_RANK
```

```
bash scripts/train_tokenizer/IBQ/run_262144.sh MASTER_ADDR MASTER_PORT NODE_RANK
```

#### 🍺 Performance and Models
| Method | #Tokens |  Codebook Size  | rFID | LPIPS  | Codebook Utilization | Checkpoint |
|:------:|:-----:|:---------------:|:----:|:------:|:---------------------:|:----:|
|IBQ| 16 $\times$ 16 |      1024       | 2.24 | 0.2580 | 99% | [IN256_Large](https://huggingface.co/TencentARC/IBQ-Tokenizer-1024/blob/main/imagenet256_1024.ckpt)|
|IBQ| 16 $\times$ 16 |      8192       | 1.87 | 0.2437 | 98% | [IN256_Large](https://huggingface.co/TencentARC/IBQ-Tokenizer-8192/blob/main/imagenet256_8192.ckpt)|
|IBQ| 16 $\times$ 16 |      16384      | 1.37 | 0.2235 | 96% | [IN256_Large](https://huggingface.co/TencentARC/IBQ-Tokenizer-16384/blob/main/imagenet256_16384.ckpt)|
|IBQ| 16 $\times$ 16 |     262144      | 1.00 | 0.2030 | 84% | [IN256_Large](https://huggingface.co/TencentARC/IBQ-Tokenizer-262144/blob/main/imagenet256_262144.ckpt)|

#### 🚀 Evaluation Scripts
```
bash scripts/evaluation/evaluation_256.sh
```

### Stage II: Training of Auto-Regressive Models

#### 🚀 Training Scripts
Please see in scripts/train_autogressive/run.sh for different model configurations.
```
bash scripts/train_autogressive/run.sh MASTER_ADDR MASTER_PORT NODE_RANK
```

#### 🚀 Sample Scripts
Please see in scripts/train_autogressive/run.sh for different sampling hyper-parameters for different scale of models.
```
bash scripts/evaluation/sample_npu.sh or scripts/evaluation/sample_gpu.sh Your_Total_Rank
```

#### 🍺 Performance and Models
| Method | Params| #Tokens | FID | IS | Checkpoint |
|:------:|:-----:|:-------:|:---:|:--:|:----------:|
|IBQ| 342M | 16 $\times$ 16 | 2.88 | 254.73 | [AR_256_B](https://huggingface.co/TencentARC/IBQ-AR-B/blob/main/AR_256_B.ckpt)|
|IBQ| 649M | 16 $\times$ 16 | 2.45 | 267.48 | [AR_256_L](https://huggingface.co/TencentARC/IBQ-AR-L/blob/main/AR_256_L.ckpt)|
|IBQ| 1.1B | 16 $\times$ 16 | 2.14 | 278.99 | [AR_256_XL](https://huggingface.co/TencentARC/IBQ-AR-XL/blob/main/AR_256_XL.ckpt)|
|IBQ| 2.1B | 16 $\times$ 16 | 2.05 | 286.73 | [AR_256_XXL](https://huggingface.co/TencentARC/IBQ-AR-XXL/blob/main/AR_256_XXL.ckpt)|
