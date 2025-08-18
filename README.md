
#  DynamicPO

> **Codes and data of DynamicPO**

## 📋 Requirements

- **Python**: 3.10.0
- **PyTorch**: 2.4.0  
- **Transformers**: 4.43.3
- **Hardware**: 4 NVIDIA A100 GPUs

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

### 3️⃣ Preference Optimization (PO)

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
