## Open-MAGVIT2: Democratizing Autoregressive Visual Generation
<p align="center">
<img src="./assets/Logo_2.png" height=300>
</p>

VQGAN remains essential in autoregressive visual generation, despite limitations in codebook size and utilization that underestimate its capabilities. [MAGVIT2](https://arxiv.org/abs/2310.05737) addresses these issues with a lookup-free technique and a large codebook ($2^{18}$), showing promising results in image and video generation, and playing a key role in [VideoPoet](https://arxiv.org/abs/2312.14125). However, we currently lack access to this tokenizer. :broken_heart:

In our codebase, we have re-implemented the MAGVIT2 tokenizer in PyTorch, closely replicating the original results. We hope our efforts will foster innovation and creativity in the field of autoregressive visual generation. :green_heart:

### 📰 News
* **[2024.06.17]** :fire::fire::fire: We release the training code of the image tokenizer and checkpoints for different resolutions, **achieving state-of-the-art performance (`0.39 rFID` for 8x downsampling)** compared to VQGAN, MaskGIT, and recent TiTok, LlamaGen, and OmniTokenizer.

### 🎤 TODOs
* [ ] Better image tokenizer with scale-up training.
* [ ] Finalize the training of the autoregressive model.
* [ ] Video tokenizer and the corresponding autoregressive model.

**🤗 Open-MAGVIT2 is still at an early stage and under active development. Stay tuned for the update!**


## 📖 Implementations

**Figure 1.** The framework of the Open-MAGVIT2 tokenizer, composed of an encoder, a lookup-free quantizer (LFQ), and a decoder.

<p align="center">
<img src="./assets/framework.png">
</p>


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


#### 🍺 Quantitative Comparison

**Table 1.** Reconstruction performance of different tokenizers on $256 \times 256$ ImageNet 50k validation set. Open-MAGVIT2 achieves SOTA results on different downsampling rates.
| Method | Token Type | #Tokens | Train Data | Codebook Size | rFID | PSNR  | Codebook Utilization | Checkpoint |
|:------:|:----:|:-----:|:-----:|:-------------:|:----:|:----:|:---------------------:|:----:|
|VQGAN | 2D | 16 $\times$ 16 | 256 $\times$ 256 ImageNet  | 1024 | 7.94 | 19.4 | - | - |
|SD-VQGAN | 2D | 16 $\times$ 16 | OpenImages | 16384 | 5.15 | - | - | - |
|MaskGIT | 2D | 16 $\times$ 16 | 256 $\times$ 256 ImageNet  | 1024 | 2.28 | - | - | -|
|LlamaGen | 2D | 16 $\times$ 16 | 256 $\times$ 256 ImageNet  | 16384 | 2.19  | 20.79 | 97% | -|
|**:fire:Open-MAGVIT2** | 2D | 16 $\times$ 16 | 256 $\times$ 256 ImageNet | 262144 | **1.53** | **21.53** | **100%** | [IN256_Base](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_256_B.ckpt)|
|ViT-VQGAN| 2D | 32 $\times$ 32 | 256 $\times$ 256 ImageNet | 8192 | 1.28 |  - | - | - |
|VQGAN | 2D | 32 $\times$ 32 | OpenImages | 16384 | 1.19 | 23.38 | - | - |
|SD-VQGAN | 2D | 32 $\times$ 32 | OpenImages | 16384 | 1.14 | - | - | - |
|OmniTokenizer-VQ| 2D | 32 $\times$ 32 | 256 $\times$ 256 ImageNet | 8192 | 1.11 | -| - | -|
|LlamaGen | 2D | 32 $\times$ 32 | 256 $\times$ 256 ImageNet | 16384 | 0.59 | 24.45 | - | - |
|**:fire:Open-MAGVIT2*** | 2D | 32 $\times$ 32 | 128 $\times$ 128 ImageNet | 262144 | **0.39** | **25.78** | **100%** |[IN128_Base](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_128_B.ckpt)|
|SD-VQGAN | 2D | 64 $\times$ 64 | OpenImages | 16384 | 0.58 | - | - | - |
|TiTok-L | 1D | 32 |  256 $\times$ 256 ImageNet | 4096 | 2.21 | - | - | - |
|TiTok-B | 1D | 64 |  256 $\times$ 256 ImageNet | 4096 | 1.70 | - | - | - | 
|TiTok-S | 1D | 128 | 256 $\times$ 256 ImageNet | 4096  | 1.71 | - | - | - |

(*) denotes that the results are from the direct inference using the model trained with $128 \times 128$ resolution without fine-tuning.

<!-- |MAGVIT2 | 16 $\times$ 16 2D token | 128 $\times$ 128 ImageNet | 262144 | 1.21 | - | - | - | - |
|Open-MAGVIT2 | 16 $\times$ 16 2D token |  128 $\times$ 128 ImageNet | 262144 | 1.56 | - | 100% | [imagenet_128_Base](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_128_B.ckpt)|  -->


**Table 2.** Compare with the original MAGVIT2 by training and testing with both $128 \times 128$ resolution as used in its original paper. ImageNet 50k validation set is used for testing.
|Method| Token Type | #Tokens | Data | LFQ | Large Codebook | Up/Down Sampler | rFID| URL | 
|:----:|:----:|:----:|:----:|:----------:|:-------:|:------:|:----------:|:------:|
|MAGVIT2 | 2D | $16 \times 16$ | 128 $\times$ 128 ImageNet | √ |  √    |   √ |1.21 | - |
|Open-MAGVIT2 | 2D | $16 \times 16$ |128 $\times$ 128 ImageNet | √ |  √ |  √ | 1.56 | [IN128_Base](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_128_B.ckpt) |


#### :eyes: Reconstruction Visualization

**Figure 2.** Visualization of the Open-MAGVIT2 tokenizer trained at $256 \times 256$ resolution and tested at $256 \times 256$ resolution (`imagenet_256_Base` version). (a) indicates the original images while (b) specifies the reconstruction images.
<p align="center">
    <img src="./assets/case.png">
</p>


**Figure 3.** Visualization of the Open-MAGVIT2 tokenizer trained at $128 \times 128$ resolution and tested at $512 \times 512$ resolution (`imagenet_128_Base` version). (a) indicates the original images while (b) specifies the reconstruction images.
<p align="center">
    <img src="./assets/case_2.png">
</p>



#### 🚀 Training Scripts
* $128\times 128$ Tokenizer Training
```
bash run_B_128.sh
```

* $256\times 256$ Tokenizer Training
```
bash run_B_256.sh
```

#### 🚀 Evaluation Scripts
* $128\times 128$ Tokenizer Evaluation
```
python evaluation.py --config_file configs/imagenet_lfqgan_128_B.yaml --ckpt_path "Your Path" --image_size 128
```

* $256\times 256$ Tokenizer Evaluation
```
python evaluation.py --config_file configs/imagenet_lfqgan_256_B.yaml --ckpt_path "Your Path" --image_size 256
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
