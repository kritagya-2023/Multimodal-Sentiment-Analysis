# **Multimodal Sentiment Analysis (MSA)**

A **multimodal sentiment analysis** project that fuses audio, vision, and text features to perform robust sentiment classification. The core architecture—**MSAModel**—leverages modality-specific LSTM/MLP encoders, private/shared encoders, attention networks, and multimodal fusion layers for comprehensive sentiment prediction.

## 🎯 About

**What is it?**  
MSA is a cutting-edge multimodal sentiment analysis framework that combines three modalities (audio, visual, and textual) to achieve superior sentiment classification performance. The architecture employs sophisticated fusion techniques and attention mechanisms to capture both modality-specific and cross-modal interactions.

**Project Status:**  
✅ Prototype / Working  
✅ Demonstrates end-to-end training, validation, and testing  

---
## ✨ **Features**

### 🧠 **Model Architecture**
- **Multimodal Encoders**  
  - *Audio branch* (`sLSTM_MLP`) - Stacked LSTM with MLP processing
  - *Vision branch* (`sLSTM_MLP`) - Visual feature encoding with temporal modeling  
  - *Text branch* (`BERT_MLP`) - BERT-style text processing with LSTM aggregation

### 🔀 **Fusion Strategy**
- **Private vs. Shared Representations**  
  - Each modality has `Private_Encoder` and `Shared_Encoder` for modality-specific and cross-modal features
  - Decoder modules for feature reconstruction

- **Modality Attention Networks (MAN)**  
  - Attention-based refinement within each modality before fusion

- **Multimodal Fusion Layers (MLF)**  
  - **Pairwise fusion:** `Audio+Visual`, `Audio+Language`, `Visual+Language`  
  - **Tri-modal fusion:** Complex `Audio+Visual+Language` combinations
  - **Hierarchical fusion:** Uni-modal → Bi-modal → Tri-modal aggregation

### 🚀 **Training & Infrastructure**
- **Custom PyTorch Dataset & DataLoader** (`MyDataset` class)
- **Comprehensive Evaluation** with accuracy and loss metrics

---

## 🛠️ Technologies & Dependencies

- **Language:** Python 3.8+  
- **Deep Learning:** PyTorch 1.10+  
- **Numerical Computing:** NumPy ≥1.19  
- **Visualization:** Matplotlib ≥3.3  
- **Data Processing:** Pickle (built-in)  

## 🔧 Installation Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/warsi1507/Multimodal-Sentiment-Analysis.git
cd Multimodal-Sentiment-Analysis
```

### 2. Create Virtual Environment (Recommended)

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (CMD):**
```bash
python -m venv venv
venv\Scripts\activate.bat
```

---

## 📁 Project Structure

```
multimodal-sentiment-analysis/
│
├── aligned_50.pkl                     # Multimodal dataset (train/valid/test splits)
├── script.py                          # Dataset utilities (inspect_dataset function)
├── MSA.ipynb                          # Complete training/evaluation pipeline
├── README.md                          # This documentation
├── input_pipeline                     
├── Best_MSA_model.pth                 # Saved model checkpoint (generated)
└──  loss_and_acc_history.png          # Training visualization (generated)
```

---
## 🔄 Data Pipeline

### Dataset Structure

The pickle dataset contains three splits with the following structure:

```python
{
    "train": {
        "id": [sample_ids...],
        "audio": numpy.array([N, time_steps, ACOUSTIC_DIM]),
        "vision": numpy.array([N, time_frames, VISUAL_DIM]), 
        "text": numpy.array([N, seq_len, LANGUAGE_DIM]),
        "classification_labels": numpy.array([N]),
        "regression_labels": numpy.array([N])  # Optional
    },
    "valid": { ... },  # Same structure
    "test": { ... }    # Same structure
}
```
### 📚 Reference: MESI (Multimodal EmotionLines Dataset)

This project leverages the **MESI (Multimodal EmotionLines Dataset)**—a widely used benchmark for multimodal sentiment and emotion analysis. MESI offers precisely aligned audio, visual, and textual features for each conversational utterance, supporting advanced research in multimodal fusion and sequence modeling.

- **Key Characteristics:**
  - Synchronized audio, visual, and text modalities for each sample
  - Fine-grained sentiment and emotion annotations at the utterance level
  - Suitable for tasks involving multimodal fusion, context modeling, and emotion recognition

- **Further Information:**
  - [MESI Dataset Paper](https://arxiv.org/abs/2106.02596)
  - [MESI Official Repository](https://github.com/declare-lab/MESI)

## 🏗️ Model Architecture

### MSAModel Components

The **MSAModel** implements a multimodal architecture:

```python
class MSAModel(nn.Module):
    def __init__(self, visual_dim, acoustic_dim, language_dim, hidden_dim, 
                 pred_hidden_dim, dropout_value, output_dim):
```

####  **Modality-Specific Encoders**
- **sLSTM_MLP**: Stacked LSTM with layer normalization for audio/visual processing
- **BERT_MLP**: BERT-style text processing with LSTM aggregation

####  **Private/Shared Encoding**
- **Private Encoders** (`E_m_p_*`): Extract modality-specific features
- **Shared Encoders** (`E_m_c_*`): Learn cross-modal representations  
- **Decoders** (`D_m_*`): Reconstruct features for consistency

####  **Attention & Fusion**
- **MAN** (Modality Attention Networks): Intra-modal attention refinement
- **MLF** (Multimodal Fusion Layers): Sophisticated inter-modal fusion
- **Hierarchical Fusion**: Uni-modal → Bi-modal → Tri-modal integration

####  **Hyperparameters**
```python
BATCH_SIZE = 16          # Training batch size
DROP = 0.4               # Dropout probability  
HID = 256                # Hidden dimensions
P_HID = 64               # Prediction layer hidden size
MAX_EPOCH = 15           # Training epochs
LEARNING_RATE = 0.0001   # Adam optimizer learning rate

# Input dimensions
VISUAL_DIM = 20          # Visual feature dimension
ACOUSTIC_DIM = 5         # Audio feature dimension  
LANGUAGE_DIM = 768       # Text feature dimension (BERT-like)
OUTPUT_DIM = 3           # Number of sentiment classes
```

#### Reference

- [A Survey on Multimodal Sentiment Analysis](https://arxiv.org/abs/2006.06250)
- [DISRFN Paper](https://www.researchgate.net/figure/The-framework-of-DISRFN-Note-Bi-LSTM-bidirectional-short-and-long-memory-network_fig1_358101693)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
---


---
## 🏋️ Training & Evaluation

### Training Pipeline

The training function includes advanced features:

- **Gradient Clipping**: Prevents gradient explosion
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Model Checkpointing**: Saves best model based on validation loss
- **Comprehensive Logging**: Tracks loss and accuracy metrics

```python
def train(model, train_loader, val_loader, optimizer, num_epochs=20):
    # Advanced training loop with:
    # - Gradient clipping (max_grad_norm = 1.0)
    # - Learning rate scheduling  
    # - Best model checkpointing
    # - Error handling for robust training
```

### Model Checkpointing

```python
# Automatic saving of best model
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(), 
    'best_val_loss': best_val_loss
}, 'Best_MSA_model.pth')
```

---

## 📊 Results & Visualization

### Training Curves

The system automatically generates `loss_and_acc_history.png` with:
- Training vs. Validation Loss curves
- Training vs. Validation Accuracy curves  
- Epoch-wise performance tracking

### Sample Output

```
Training model from scratch.
Epoch 1/15 | Train Loss: 0.8955, Val Loss: 0.8980 | Train Acc: 52.57%, Val Acc: 54.15%
Epoch 2/15 | Train Loss: 0.8814, Val Loss: 0.8943 | Train Acc: 52.88%, Val Acc: 54.15%
Epoch 3/15 | Train Loss: 0.8774, Val Loss: 0.8909 | Train Acc: 52.88%, Val Acc: 54.15%
...
Average Test Loss: 0.7478
Test Accuracy: 76.09%
```


## 🤝 Contribution
- [@samar warsi](https://github.com/warsi1507)
- [@kritagya](https://github.com/kritagya-2023)
- [@kavya_kumar_agrawal](https://github.com/Kavya-Agrawal)

### 🙏 **Acknowledgments**
- Original dataset creators and researchers
- PyTorch community for comprehensive documentation
- Contributors to multimodal learning research
- Open-source community for inspiration and support