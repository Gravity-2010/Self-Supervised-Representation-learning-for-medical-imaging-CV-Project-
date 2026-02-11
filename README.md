# Self-Supervised Representation Learning for Medical Imaging

A comprehensive project exploring self-supervised learning techniques, specifically SimCLR (Simple Framework for Contrastive Learning of Visual Representations), applied to chest X-ray analysis. This work investigates medical-specific data augmentations, label efficiency, cross-dataset transfer learning, and model interpretability for thoracic disease detection.

## 🎯 Project Overview

This project addresses the challenge of limited labeled data in medical imaging by leveraging self-supervised learning. We implement and evaluate SimCLR on chest X-ray datasets, developing domain-specific augmentation strategies and analyzing model performance across different data scenarios.

### Key Features

- **Self-Supervised Learning**: Implementation of SimCLR framework for learning representations from unlabeled chest X-rays
- **Medical-Specific Augmentations**: Custom data augmentation pipeline tailored for medical imaging
- **Label Efficiency Analysis**: Evaluation of model performance with varying amounts of labeled data
- **Cross-Dataset Transfer Learning**: Investigation of knowledge transfer between NIH ChestX-ray14 and CheXpert datasets
- **Model Interpretability**: Grad-CAM visualizations and embedding analysis for understanding model decisions
- **Multi-Label Classification**: Support for detecting multiple thoracic diseases simultaneously

## 📊 Datasets

### CheXpert
- **Source**: [Kaggle - CheXpert Dataset](https://www.kaggle.com/datasets/ashery/chexpert?select=train)
- **Description**: Large dataset of chest radiographs with multi-label annotations
- **Classes**: 14 observations including No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices

### NIH ChestX-ray14
- **Source**: [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **Description**: Dataset containing over 100,000 frontal-view X-ray images
- **Classes**: 14 disease labels including Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, and more

## 🗂️ Repository Structure

```
├── 3Models.ipynb                    # Implementation of three model variants/experiments
├── CheXpertResNet_(1).ipynb        # ResNet training on CheXpert dataset
├── Transfer_Learning.ipynb          # Cross-dataset transfer learning experiments
└── README.md                        # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
torch >= 1.9.0
torchvision >= 0.10.0
numpy
pandas
matplotlib
scikit-learn
opencv-python
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Gravity-2010/Self-Supervised-Representation-learning-for-medical-imaging-CV-Project-.git
cd Self-Supervised-Representation-learning-for-medical-imaging-CV-Project-
```

2. Install required packages:
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn opencv-python pillow
```

3. Download datasets:
   - CheXpert: Download from [Kaggle](https://www.kaggle.com/datasets/ashery/chexpert?select=train)
   - NIH ChestX-ray14: Download from [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC)

### Usage

#### 1. Training Models
Open and run the notebooks in the following order:

**3Models.ipynb**: Experiment with different model architectures and configurations
```bash
jupyter notebook 3Models.ipynb
```

**CheXpertResNet_(1).ipynb**: Train ResNet on CheXpert dataset
```bash
jupyter notebook CheXpertResNet_\(1\).ipynb
```

**Transfer_Learning.ipynb**: Perform cross-dataset transfer learning
```bash
jupyter notebook Transfer_Learning.ipynb
```

## 🔬 Methodology

### Self-Supervised Pre-training (SimCLR)

1. **Contrastive Learning Framework**: Learn representations by maximizing agreement between differently augmented views of the same image
2. **Medical-Specific Augmentations**: 
   - Random rotation (small angles to preserve anatomical orientation)
   - Random horizontal flipping
   - Color jittering (brightness, contrast adjustments)
   - Gaussian blur
   - Random cropping and resizing
   - Careful consideration of medical imaging constraints

3. **Architecture**: ResNet-based encoder with projection head for contrastive learning

### Supervised Fine-tuning

After pre-training, models are fine-tuned on labeled data for disease classification tasks with varying amounts of labeled samples to evaluate label efficiency.

### Transfer Learning

Evaluation of model performance when:
- Pre-trained on NIH, fine-tuned on CheXpert
- Pre-trained on CheXpert, fine-tuned on NIH
- Comparison with models trained from scratch

## 📈 Experiments & Results

### Key Analyses

1. **Label Efficiency**: Performance comparison using 1%, 10%, 25%, 50%, and 100% of labeled data
2. **Cross-Dataset Generalization**: Transfer learning effectiveness between NIH and CheXpert
3. **Augmentation Impact**: Ablation studies on different augmentation strategies
4. **Interpretability**: Grad-CAM visualizations showing which regions the model focuses on

### Evaluation Metrics

- AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
- Accuracy
- Precision, Recall, F1-Score
- Per-class performance analysis

## 🔍 Model Interpretability

### Grad-CAM Visualizations
- Visual explanations highlighting important regions for predictions
- Validation of model attention on clinically relevant areas
- Comparison of attention maps across different training strategies

### Embedding Analysis
- t-SNE and UMAP visualizations of learned representations
- Cluster analysis to understand feature space organization
- Comparison of embeddings from different training paradigms

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- Bug fixes
- New features
- Documentation improvements
- Additional experiments or analyses

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{selfsupervised_medical_imaging,
  author = {Gravity-2010},
  title = {Self-Supervised Representation Learning for Medical Imaging},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Gravity-2010/Self-Supervised-Representation-learning-for-medical-imaging-CV-Project-}
}
```

## 📚 References

- **SimCLR**: Chen, T., et al. "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020.
- **CheXpert**: Irvin, J., et al. "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels." AAAI 2019.
- **NIH ChestX-ray14**: Wang, X., et al. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks." CVPR 2017.

## ⚠️ Disclaimer

This project is for research and educational purposes only. The models and analyses are not intended for clinical diagnosis or medical decision-making. Always consult qualified healthcare professionals for medical advice.

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Project Status**: Active Development

**Last Updated**: December 2025
