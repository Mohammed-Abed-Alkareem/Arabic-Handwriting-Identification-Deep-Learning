# Arabic Handwriting Identification Using Deep Learning

## Overview
This project focuses on optimizing various convolutional neural network (CNN) architectures for Arabic handwriting identification using the **AHAWP (Arabic Handwritten Automatic Word Processing) Dataset**. The study includes designing a custom network, applying data augmentation, and implementing transfer learning.

The primary objectives include:
- Designing and training custom CNN architectures.
- Implementing data augmentation techniques.
- Training standard CNN architectures (ResNet-18, ResNet-50, MobileNet, and Shallow ResNet).
- Fine-tuning pretrained models to leverage prior knowledge from similar tasks.
- Comparing model performances based on accuracy and efficiency.

## Dataset
The **AHAWP Dataset** consists of **8,144 word-level images** of **10 unique words** handwritten by **82 individuals**, with **10 samples per word**. This dataset is well-suited for evaluating **local feature extraction algorithms and deep learning models**.

### **Preprocessing & Augmentation**
- Images resized to **224×224 pixels**.
- Grayscale conversion for intensity-based recognition.
- Pixel values **normalized** (mean=0.5, std=0.5).
- Dataset split into **training (64%), validation (16%), and testing (20%)**.
- **Data augmentation techniques**: rotation, scaling, illumination adjustment.



## Model Implementations
### **1. Custom CNN Architectures**
- Designed multiple CNN architectures with varying depths and layers.
- Experimented with **Batch Normalization, Dropout, Global Average Pooling (GAP), and different activation functions**.
- Implemented optimizers **Adam and AdamW**, where **AdamW performed better**.

### **Optimizer Comparison**
The following optimizers were tested:
- **Adam Optimizer:**
  ```python
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=0.001)
  ```
- **AdamW Optimizer (Best Performance):**
  ```python
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.AdamW(net.parameters(), lr=0.002, weight_decay=1e-2)
  ```

### **2. Data Augmentation Impact**
- Without augmentation, **NetCNN\_4** achieved **35.5% validation accuracy**.
- With augmentation, accuracy improved to **73.2%**, highlighting the effectiveness of augmentation in generalization.

### **3. Standard CNN Architectures**
- **MobileNet:** Lightweight model optimized for efficiency.
- **ResNet-18:** Residual network with 18 layers.
- **ResNet-50:** Deeper residual network with 50 layers.
- **Shallow ResNet:** Custom shallow architecture leveraging residual connections.

### **4. Transfer Learning & Fine-Tuning**
- **Fine-tuned pretrained models (MobileNet, ResNet-50) on AHAWP dataset**.
- **Pretrained MobileNet achieved 80.7% validation accuracy**, outperforming the untrained version.
- **Pretrained ResNet-50 underperformed (40.5% validation accuracy)** due to overfitting.

## Performance Comparison
| Model                 | Training Accuracy | Validation Accuracy |
|----------------------|------------------|------------------|
| **NetCNN\_4**        | 59.9%            | 55%             |
| **MobileNet**        | 87.6%            | 71.2%           |
| **ResNet-18**        | 96.2%            | 77.5%           |
| **ResNet-50**        | 90.7%            | 63.6%           |
| **Pretrained ResNet-50** | 92.1%       | 40.5%           |
| **Pretrained MobileNet** | 92.9%       | 80.7%           |
| **Shallow ResNet**   | 98.2%            | 94.0%           |

## Conclusion
- **Shallow ResNet outperformed all models**, achieving **94% validation accuracy**.
- **Pretrained MobileNet demonstrated strong generalization**.
- **Fine-tuning boosted performance in most cases, but overfitting was observed in deep networks like ResNet-50**.
- **Data augmentation significantly improved model performance**, emphasizing the importance of dataset enhancement.
- **AdamW optimizer performed better than Adam** in optimizing network training.

