# Hybrid Deep Learning Architectures for Diabetic Retinopathy Detection

A deep learning-based approach to detect and classify **Diabetic Retinopathy (DR)** using multiple architectures:  
- **Custom Convolutional Neural Networks (CNN)**  
- **VGG16** with transfer learning  
- **Vision Transformers (ViT)** for global attention-based classification

This project explores the comparative performance of these models on fundus image data, focusing on accuracy, interpretability (Grad-CAM), and generalization.

---

## 🚀 What This Repository Contains

- **Preprocessing and Augmentation** of fundus images
- **Model Training Pipelines** for CNN, VGG16, and ViT
- **Evaluation Scripts**: accuracy, confusion matrix, ROC-AUC
- **Visualization Tools**: training curves, Grad-CAM heatmaps

---

## 🔬 Key Highlights

- 🧠 Vision Transformer showed fast convergence and high training accuracy  
- ⚖️ VGG16 balanced performance with pre-trained weights  
- 💡 CNN model provided excellent parameter efficiency  
- 🎯 Grad-CAM visualizations offered interpretability of the model predictions

---

## 🧾 Paper Publication

> This research work has been officially **accepted and published** in the  
> [International Journal of Science, Engineering and Technology (IJSET)](https://www.ijset.in)

- **Paper Title:** Hybrid Deep Learning Architectures for Diabetic Retinopathy Detection  
- **Status:** Accepted for online publication  
- **ISSN (Online):** 2348-4098  
- **Paper ID:** IJSET_V13I2_15889  

---

## 📂 GitHub Repository

Explore the code and experiments in the GitHub repo:  
🔗 [github.com/garg-shiv/Deep-Learning-for-Diabetic-Retinopathy-Detection-Study-of-CNNs-VGG16-and-Vision-Transformers](https://github.com/garg-shiv/Deep-Learning-for-Diabetic-Retinopathy-Detection-Study-of-CNNs-VGG16-and-Vision-Transformers)

---

## ✅ Quick Start

```bash
# Clone the repository
git clone https://github.com/garg-shiv/Deep-Learning-for-Diabetic-Retinopathy-Detection-Study-of-CNNs-VGG16-and-Vision-Transformers.git

# Install dependencies
pip install -r requirements.txt

# Train a model (example)
python src/train.py --model vgg16
```

---

## 🧠 Built With

- PyTorch
- Timm (for Vision Transformers)
- Grad-CAM
- Matplotlib, Scikit-learn
