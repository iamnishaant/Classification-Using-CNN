# Flower Classification Using CNN (Oxford 102 Flower Dataset)

## 📌 Project Overview
This project performs flower classification using a convolutional neural network (MobileNetV2) trained on the Oxford 102 Flower Category dataset. The goal is to accurately classify flower images into 102 classes using transfer learning and fine-tuning.

## 📁 Dataset
- **Images**: 102 flower categories with at least 40 images each.
- **Files**:
  - `102flowers.tgz` - Raw images
  - `imagelabels.mat` - Image labels
  - `setid.mat` - Train/val/test splits
  - `cat_to_name.json` - Maps category IDs to flower names

## 🧠 Model Architecture
- Base Model: `MobileNetV2` (pre-trained on ImageNet)
- Custom head:
  - GlobalAveragePooling
  - Dense (512 units, ReLU)
  - Output Dense layer (102 classes, softmax)

## 🏋️ Training
- Two-phase training:
  1. Train top classifier (base frozen)
  2. Fine-tune entire model (last 50 layers trainable)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 10 + 5 fine-tuning

## 📊 Evaluation
- Test Accuracy: **73% **
- Classification Report
- Confusion Matrix (Partial: First 20 classes)
- Per-Class Accuracy Bar Chart

## 📈 Visuals
- Confusion Matrix heatmap
- Per-Class Accuracy plot
- Grad-CAM visualizations

## 💾 Saved Model
- `flower102_mobilenet_finetuned_model.h5`


