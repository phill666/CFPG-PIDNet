# CFPG-PIDNet: Confidence-Feedback Peripheral Gating for Collapse-Resilient Industrial Surface Defect Segmentation and Geometric Measurement

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange) ![License](https://img.shields.io/badge/License-MIT-green)

**Tianyu Liu\*, Haiyu Li\*, Siqi Ran, Xiangyu Wang**
\* Co-corresponding authors: Tianyu Liu and Haiyu Li. Contact: **liuty@mail.xzcit.cn**.

Official manuscript implementation. CFPG occupies the DFM output, with a learnable identity-preserving residual scale initialized to 0.1. Native P/I/D branches are retained. Inference returns 1/8-resolution logits; evaluation interpolates logits to each original image size before softmax and measurement.

## Highlights

- **+59,504 parameters (+0.78%)**, **7.683 M** total, **+0.206 GMACs** at 512x512.
- KSDD2 four-seed stress test (42,1024,2026,3407): **0/4 collapses** versus PIDNet **2/4** in the manuscript failure audit, **+10.95 pp** mean dIoU, **96.1% lower seed variance**.
- Best evaluated three-dataset MeasurementOpt macro **BDE 25.10 px** (**47.4%** below PIDNet) and **Spearman rho 0.832**.
- RTX 4090, true FP32, 512x512, batch 1: **164.06 FPS eager**, **187.45 FPS CUDA Graph**.

## Core results

Macro means first average seeds 42/2026/3407 within each dataset, then equally average KSDD2, MTD and NEU Seg under MeasurementOpt.

| Model | dIoU (%) | Dice | RAE | LE | BDE (px) | CE (px) | ConnE | Spearman rho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PIDNet | 61.85 | 0.7595 | 0.5971 | 0.8371 | 47.73 | 54.19 | 1.543 | 0.754 |
| CFPG-PIDNet | **66.37** | **0.7967** | **0.5357** | **0.6559** | **25.10** | **31.36** | **1.253** | **0.832** |

| KSDD2 four-seed stress test | Mean dIoU (%) | Sample std (pp) | Collapse audit |
|---|---:|---:|---:|
| PIDNet | 50.47 | 13.98 | 2/4 |
| CFPG-PIDNet | **61.42** | **2.77** | **0/4** |

Collapse counts refer to the manuscript audit. The evaluator independently reports exact empty predictions and Empty Prediction Rate; it does not invent a new collapse cutoff.

## Installation

Tested with Python 3.10, PyTorch 2.0.0, CUDA 11.8 and RTX 4090.

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python selfcheck.py
```

## Public datasets and fixed splits

Download from [KSDD2](https://www.vicos.si/resources/kolektorsdd2/), [Magnetic Tile](https://github.com/abin24/Magnetic-tile-defect-datasets), and the [NEU Seg segmentation release](https://data.mendeley.com/datasets/z8g47vzt4p/1). Their licenses and original citations apply. No images, annotations or checkpoints are bundled.

```text
data/
  KSDD2/
    train/  # image.png and image_GT.png
    test/   # image.png and image_GT.png
  MTD/
    MT_Blowhole/Imgs/  # RGB image plus same-stem binary PNG mask
    MT_Break/Imgs/
    MT_Crack/Imgs/
    MT_Fray/Imgs/
    MT_Free/Imgs/      # absent free-class masks are interpreted as zeros
    MT_Uneven/Imgs/
  NEU_Seg/
    images/training/       # sample.jpg
    annotations/training/  # sample.png
```

Committed `splits/` lists define the canonical train/validation/test samples. Keep the lists fixed across seeds. Use `--data-root` to override location. Restore exact original names, including KSDD2 copied training-pair names in the split list. Inputs use RGB, PIL bilinear 512x512 resizing and ImageNet mean/std normalization.

## Training and internal ablations

From scratch: 160 epochs, batch 16, SGD lr=0.01, momentum=0.9, weight decay=0.0005, poly power=0.9. Loss is main CE+Dice plus 0.4 auxiliary semantic CE+Dice. Training uses AMP. No added boundary loss is used; inference omits auxiliary heads.

```powershell
python train.py --config configs/ksdd2.yaml --variant cfpg --seeds 42 2026 3407
python train.py --config configs/ksdd2.yaml --variant baseline --seeds 42 1024 2026 3407
python train.py --config configs/mtd.yaml --variant cfpg --seeds 42 2026 3407
python train.py --config configs/neu_seg.yaml --variant cfpg --seeds 42 2026 3407
powershell -File scripts/reproduce.ps1
powershell -File scripts/ablations.ps1
```

| Variant | Peripheral/coarse gate | Center suppression | Entropy feedback | GRN | Added parameters |
|---|---|---|---|---|---:|
| ctrl_conv | 3x3-BN-GELU-1x1 control | - | - | - | 56,537 |
| cfpg_periph_only | yes | no | no | no | 55,060 |
| cfpg_plus_csa | yes | yes | no | no | 58,900 |
| cfpg_plus_entropy_no_grn | yes | yes | yes | no | 59,248 |
| cfpg | yes | yes | yes | yes | 59,504 |

Strip kernels are (7,15,31). Physical priors combine fixed Sobel/Laplacian filters and local variance. Coarse-only stages retain SE reweighting and output projection; disabled feedback branches are removed. All mechanisms share the DFM-output position and 0.1 initial residual scale. Shared initial tensors and subsequent RNG state are preserved.

## Validation calibration and native-resolution measurement

```powershell
python eval.py --config configs/ksdd2.yaml --variant cfpg --weights checkpoints/ksdd2/cfpg_seed42.pth --output results/eval/ksdd2
```

The locked validation threshold grid is **0.30,0.35,...,0.80**. Argmax uses 0.50. DiceOpt maximizes validation Dice. MeasurementOpt retains at least 95% of best validation Dice, then minimizes RAE+BDE+CE. Thresholds are fixed before testing.

Metrics include mIoU/dIoU/Dice/Precision/Recall, relative area error RAE, skeleton length error LE, bidirectional boundary distance BDE, centroid error CE, component error ConnE, Empty Prediction Rate, and GT-empty Image/Pixel FPR and mean predicted area. Empty positive predictions receive the original diagonal for BDE/CE. Undefined Spearman stays null.

Severity bounds and tertiles are fitted only to positive training GT. The continuous score uses area/length/irregularity weights 0.5/0.3/0.2, irregularity P^2/(4*pi*A), and no prediction clipping.

```powershell
python bootstrap_ci.py --ours results/paired_ours.json --other results/paired_baseline.json --output results/bootstrap.json
```

Input JSONs contain identical `seeds` and `sample_ids`, and `values` of shape [seed, positive image, 7], ordered TP/FP/FN/RAE/LE/BDE/CE. 10,000 paired percentile resamples match images and use seed hierarchy for multiseed data. dIoU is mean seed-level positive-image micro IoU. Differences are ours minus comparator; intervals are unadjusted for multiple comparisons.

## Uniform complexity and dual-mode speed

```powershell
python benchmark_speed.py --model cfpg --output results/cfpg_speed.json
python benchmark_speed.py --model stdcseg --output results/stdc_speed.json
```

Models: `unet`, `bisenetv2`, `ddrnet23slim`, `stdcseg`, `segformer_b0`, `ppliteseg_stdc1`, `baseline`, `cfpg`. Baseline definitions are included for inference; upstream training recipes and licenses apply. STDC skips unused detail/auxiliary heads and backbone classification parameters. Use a separate process per model. True FP32, TF32 disabled, eval/inference_mode, 100 warmups and 1000 CUDA-event iterations; cuDNN autotuning is enabled before the first forward. Inputs remain GPU-resident. THOP counts each physical module once; unsupported functional ops are excluded. GFLOPs=2*GMACs.

## Fig. 4-6 publication visualizations

```powershell
python visualize.py --manifest configs/visualization_manifest.json --output results/figures
```

Provide local seed42 weights at the configurable manifest paths. Supplied MeasurementOpt thresholds and nine specimen IDs match the final figure protocol. Fig.4 uses true hooked tensors, percentile display scaling and raw ranges. Fig.5 shows six successes, Fig.6 three failures. PNG/PDF export uses equal 4:3 frames at 300 DPI. Union-centred, GT-protected crops affect display only; annotated Dice/RAE/BDE use complete native images. The frozen MTD Break ID is `exp1_num_98080`.

## Source audit and licensing

`selfcheck.py` compiles every Python source, validates forward shapes and exact parameter increments, then audits paths, sensitive patterns and repository size. Root MIT: Copyright (c) 2026 Tianyu Liu. Third-party code retains separate licenses, including GPL-3.0 U-Net components and the NVIDIA SegFormer reference license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@article{liu_cfpg_pidnet,
  title={CFPG-PIDNet: Confidence-Feedback Peripheral Gating for Collapse-Resilient Industrial Surface Defect Segmentation and Geometric Measurement},
  author={Liu, Tianyu and Li, Haiyu and Ran, Siqi and Wang, Xiangyu},
  journal={To be updated upon publication},
  year={2026}
}
```

## Acknowledgments

We thank PIDNet, U-Net, BiSeNetV2, DDRNet, STDC-Seg, SegFormer and PP-LiteSeg communities and dataset publishers. Original author headers and upstream license texts are retained in `third_party_licenses/`.
