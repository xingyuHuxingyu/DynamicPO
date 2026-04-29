
#  DynamicPO

## 📢 Status

We are currently refining the codebase for better readability and elegance, carefully verifying environment dependencies for improved reproducibility, and preparing additional implementation details that could not be included in the paper due to space limitations.
Please stay tuned — the finalized version will be released soon.

> **Official Implementation of "DynamicPO: Dynamic Preference Optimization for Recommendation".**

## Multi-negative Objective Functions

### DMPO

```math
\mathcal{L}_{\mathrm{DMPO}}(\pi_\theta; \pi_{\mathrm{ref}})
=
\mathbb{E}_{(x_u, y_w, y_l) \sim \mathcal{D}}
\left[
\log \sigma
\left(
\beta \log
\frac{\pi_\theta(y_w \mid x_u)}
{\pi_{\mathrm{ref}}(y_w \mid x_u)}
-
\frac{1}{k}
\sum_{i=1}^{k}
\beta \log
\frac{\pi_\theta(y_i \mid x_u)}
{\pi_{\mathrm{ref}}(y_i \mid x_u)}
\right)
\right]
```

### MPPO

```math
\mathcal{L}_{\mathrm{MPPO}}(\pi_\theta)
=
-
\mathbb{E}
\left[
\log \sigma
\left(
N \cdot \pi_\theta(y_w \mid x)
-
\sum_{i=1}^{N}
\pi_\theta(y_{l_i} \mid x)
\right)
\right]
```

### S-DPO

```math
\mathcal{L}_{\mathrm{S-DPO}}(\pi_\theta; \pi_{\mathrm{ref}})
=
-
\mathbb{E}_{(x_u, e_p, E_d) \sim \mathcal{D}}
\left[
\log
\left(
-
\log
\sum_{e_d \in E_d}
\exp
\left(
\beta \log
\frac{\pi_\theta(e_d \mid x_u)}
{\pi_{\mathrm{ref}}(e_d \mid x_u)}
-
\beta \log
\frac{\pi_\theta(e_p \mid x_u)}
{\pi_{\mathrm{ref}}(e_p \mid x_u)}
\right)
\right)
\right]
```


## 📋 Requirements
You can refer to the `requirements.txt`, or install the core modules with:

- **Python**: `python>=3.9`
- **PyTorch**: `2.4.0`
- **Transformers**: `4.43.3`
- **Hardware**: 4 NVIDIA A100 GPUs

To install using pip, run:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1️⃣ Data Preparation

Extract the data files to prepare LastFM dataset:

```bash
cd ./data
unzip lastfm-sft-cans20.zip
```

This will generate the train, validation, and test datasets for LastFM.

### 2️⃣ Supervised Fine-Tuning (SFT)

Start the SFT training process:

```bash
sh ./scripts/sft.sh &
```

### 3️⃣ Multi-Negative Preference Optimization (DMPO-based DynamicPO)

Run the DynamicPO training:

```bash
sh ./scripts/DynamicPO_DMPO.sh &
```

### 4️⃣ Model Inference

Perform inference with the trained model:

```bash
sh ./scripts/inference.sh &
```

## 📁 Project Structure

```
DynamicPO/
├── data/           # Dataset files
├── scripts/        # Training and inference scripts        
└── trainer/        #  Model implementations
```

## 🙏 Acknowledgment
Our implementation is built upon the [TRL library](https://github.com/huggingface/trl). We are grateful to the authors of [DMPO](https://github.com/BZX667/DMPO),MPPO and [S-DPO](https://github.com/chenyuxin1999/S-DPO) for their insightful work on multi-negative preference optimization for recommendation systems, which inspired parts of this work.
