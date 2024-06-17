## Open-MAGVIT2: Democratizing Autoregressive Visual Generation
<p align="center">
<img src="./assets/Logo_2.png" height=300>
</p>
Until now, VQGAN, the initial tokenizer is still acting an indispensible role in mainstream tasks, especially autoregressive visual generation. Limited by the bottleneck of the size of codebook and the utilization of code, the capability of AR generation with VQGAN is underestimated.

Therefore, [MAGVIT2](https://arxiv.org/abs/2310.05737) proposes a powerful tokenizer for visual generation task, which introduces a novel LookUpFree technique when quantization and extends the size of codebook to $2^{18}$, exhibiting promising performance in both image and video generation tasks. And it plays an important role in the recent state-of-the-art AR video generation model [VideoPoet](https://arxiv.org/abs/2312.14125). However, we have no access to this strong tokenizer so far. :frowning_face:

In the codebase, we follow the significant insights of tokenizer design in MAGVIT-2 and re-implement it with Pytorch, achieving the closest results to the original so far. We hope that our effort can foster innovation and creativity within the field of Autoregressive Visual Generation. :smile:

### 📰 News
* **[2024.06.17]** 🤗 We released the training code of the image tokenizer and checkpoints for different resolutions.

### 🎤 TODOs
* [ ] Better image tokenizer with scale-up training.
* [ ] Finalize the training of autoregressive model.
* [ ] Video tokenizer and the corresponding autoregressive model.

**Open-MAGVIT2 is still at an early stage and under active development. Stay tuned for the update!**


## 📖 Implementations

<p align="center">
<img src="./assets/framework.png">
</p>

Figure 1. The framework of Open-MAGVIT2 tokenizer which consists of a Encoder, LookupFree Quantizer and a Decoder.

### 🛠️ Installation
- **Env**: We have tested on `Python 3.8.8` and `CUDA 11.7` (other versions may also be fine).
- **Dependencies**: `pip install -r requirements`
- **Datasets**
```
imagenet
└── train/
    ├── n01440764
        ├── n01440764_10026.JPEG
        ├── n01440764_10027.JPEG
        ├── ...
    ├── n01443537
    ├── ...
└── val/
    ├── ...
```

### Stage I: Training of Visual Tokenizer
<!-- * `Stage I Tokenizer Training`: -->
We follow the design of Generator in MAGVIT-2 but use PatchGAN instead of StyleGAN as Discriminator for GAN training. We use the combination of Loss utilized in MAGVIT-2 and VQGAN for better training stability and reconstruction quality. All the training details can be found in the config files. Note that, we train our model using 32 $\times$ V100.


#### Reconstruction Visualization
<p align="center">
    <img src="./assets/case.png">
</p>

Figure 2. Visualization of the tokenizer trained with $256 \times 256$ ImageNet reconstruction ability in $256 \times 256$ resolution. (a) indicates the original images while (b) specifies the reconstruction images.

<p align="center">
    <img src="./assets/case_2.png">
</p>

Figure 3. Visualization of the tokenizer trained with $128 \times 128$ reconstruction ability in $512 \times 512$ resolution. (a) indicates the original images while (b) specifies the reconstruction images.

#### 🍺 Performance Comparison
| Method | Token | Data | Codebook Size | rFID | PSNR  | Codebook Utilization | URL |
|:------:|:-----:|:-----:|:-------------:|:----:|:----:|:---------------------:|:----:|
|VQGAN |  16 $\times$ 16 2D token | 256 $\times$ 256 ImageNet  | 1024 | 7.94 | 19.4 | - | - |
|MaskGIT | 16 $\times$ 16 2D token | 256 $\times$ 256 ImageNet  | 1024 | 2.28 | - | - | -|
|LlamaGen | 16 $\times$ 16 2D token | 256 $\times$ 256 ImageNet  | 16384 | 2.19  | 20.79 | 97% | -|
|TiTok-L |  32 1D token |  256 $\times$ 256 ImageNet | 4096 | 2.21 | - | - | - |
|TiTok-B |  64 1D token |  256 $\times$ 256 ImageNet | 4096 | 1.70 | - | - | - | 
|TiTok-S | 128 1D token | 256 $\times$ 256 ImageNet | 4096  | 1.71 | - | - | - |
|Open-MAGVIT2 | 16 $\times$ 16 2D token | 256 $\times$ 256 ImageNet | 262144 | 1.53 | 21.53 | 100% | [imagenet_256_Base](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_256_B.ckpt)|
|VQGAN | 32 $\times$ 32 2D token | OpenImages | 16384 | 1.19 | 23.38 | - | - |
|ViT-VQGAN| 32 $\times$ 32 2D token | 256 $\times$ 256 ImageNet | 8192 | 1.28 |  - | - | - |
|OmniTokenizer-VQ| 32 $\times$ 32 2D token | 256 $\times$ 256 ImageNet | 8192 | 1.11 | -| - | -|
|LlamaGen | 32 $\times$ 32 2D token | 256 $\times$ 256 ImageNet | 16384 | 0.59 | 24.45 | - | - |
|Open-MAGVIT2* | 32 $\times$ 32 2D token | 128 $\times$ 128 ImageNet | 262144 | 0.39 | 25.78 | 100% |[imagenet_128_Base](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_128_B.ckpt)|
<!-- |MAGVIT2 | 16 $\times$ 16 2D token | 128 $\times$ 128 ImageNet | 262144 | 1.21 | - | - | - | - |
|Open-MAGVIT2 | 16 $\times$ 16 2D token |  128 $\times$ 128 ImageNet | 262144 | 1.56 | - | 100% | [imagenet_128_Base](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_128_B.ckpt)|  -->

Table 1. Tokenization comparison on $256 \times 256$ ImageNet 50k validation. * denotes that the results are from the direct inference using the model trained with $128 \times 128$ resolution without fine-tuning.

|Method| Token | Data | LFQ | Large Codebook | Up/DownSampler | rFID| URL | 
|:----:|:----:|:----:|:----------:|:-------:|:------:|:----------:|:------:|
|MAGVIT2 | $16 \times 16$ 2D token | 128 $\times$ 128 ImageNet | √ |  √    |   √ |1.21 | - |
|Open-MAGVIT2 | $16 \times 16$ 2D token |128 $\times$ 128 ImageNet | √ |  √ |  √ | 1.56 | [imagenet_128_Base](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_128_B.ckpt) |

Table 2. Tokenization comparison on $128 \times 128$ ImageNet 50k validation with MAGVIT2.

#### 🚀 Training Scripts
* $128\times 128$ Tokenizer Training
```
bash run_B_128.sh
```

* $256\times 256$ Tokenizer Training
```
bash run_B_256.sh
```

### Stage II: Training of Autoregressive Generation
<!-- * `Stage II AutoRegressive Training`: -->
MAGVIT2 utilizes Non-AutoRegressive transformer for image generation. Instead, we would like to exploit the potential of Autogressive Visual Generation with the relatively large codebook. We are currently exploring Stage II training.


## ❤️ Acknowledgement
We thank [Lijun Yu](https://me.lj-y.com/) for his encouraging discussions. We refer a lot from [VQGAN](https://github.com/CompVis/taming-transformers) and [MAGVIT](https://github.com/google-research/magvit). Thanks for their wonderful work.

## ✏️ Citation
If you found the codebase helpful, please cite it.
```
@software{Luo_Open-MAGVIT2_2024,
author = {Luo, Zhuoyan and Shi, Fengyuan and Ge, Yixiao},
month = jun,
title = {{Open-MAGVIT2}},
url = {https://github.com/TencentARC/Open-MAGVIT2},
version = {1.0},
year = {2024}
}

@inproceedings{
yu2024language,
title={Language Model Beats Diffusion - Tokenizer is key to visual generation},
author={Lijun Yu and Jose Lezama and Nitesh Bharadwaj Gundavarapu and Luca Versari and Kihyuk Sohn and David Minnen and Yong Cheng and Agrim Gupta and Xiuye Gu and Alexander G Hauptmann and Boqing Gong and Ming-Hsuan Yang and Irfan Essa and David A Ross and Lu Jiang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=gzqrANCF4g}
}
```
