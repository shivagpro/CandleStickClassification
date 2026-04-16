# Candle Orientation Classification with Deep Learning

## Project Overview
This project classifies candle images by orientation (**Up vs Down**) using a fine-tuned ResNet18 deep learning model trained on PyTorch.

---

## Dataset
- **Source**: [Kaggle - Candle Image Data](https://www.kaggle.com/datasets/raimiazeezbabatunde/candle-image-data)
- **Structure**:
```
candle-image-data/
├── Train/
│   ├── Down/
│   └── Up/
└── Test/
    ├── Down/
    └── Up/
```

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip or conda
- GPU (optional but recommended)

### Installation

1. Clone or download the project

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
- Visit [Kaggle - Candle Image Data](https://www.kaggle.com/datasets/raimiazeezbabatunde/candle-image-data)
- Download and extract to a local directory

---

## Running the Analysis

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `candle_classification.ipynb`

3. Update the `base_dir` variable in Section 2:
```python
base_dir = 'path/to/candle-image-data'
```

4. Run all cells sequentially (**Kernel → Run All**)

---

## Project Structure
```
├── README.md
├── requirements.txt
└── candle_classification.ipynb
```

---

## Notebook Sections
- Library Imports — Dependencies and GPU setup  
- Data Paths — Define train/test and up/down folder locations  
- Data Exploration — Count and analyze images per folder  
- Sample Visualization — View representative candle images  
- Custom Dataset Class — PyTorch Dataset for loading images  
- Data Preprocessing — Image normalization and augmentation  
- Model Definition — ResNet18 transfer learning setup  
- Training Setup — Loss function, optimizer, and training loop  
- Model Training — Train for 15 epochs with validation  
- Training History — Plot loss and accuracy curves  
- Model Evaluation — Calculate precision, recall, F1-score  
- Confusion Matrix — Visualize per-class performance  
- Prediction Visualization — Display sample predictions with labels  
- Results Summary — Analysis and recommendations  

---

## Model Architecture
- **Base Model**: ResNet18 (pre-trained on ImageNet)  
- **Fine-tuning Strategy**: Freeze early layers, train final fully connected layer  
- **Output Classes**: 2 (Down = 0, Up = 1)  
- **Input Size**: 224 × 224 pixels  

---

## Training Configuration
- **Batch Size**: 32  
- **Epochs**: 15  
- **Learning Rate**: 0.001  
- **Optimizer**: Adam  
- **Loss Function**: CrossEntropyLoss  
- **Learning Rate Scheduler**: StepLR (step_size = 5, gamma = 0.1)  

---

## Data Augmentation
Applied during training:
- Random horizontal flip (50% probability)  
- Random rotation (±15 degrees)  
- Color jitter (brightness and contrast variation)  
- Resize to 224 × 224  
- Normalization using ImageNet statistics  

---

## Results

### Test Set Metrics
- **Accuracy**: 53.56%  
- **Precision**: 0.5161  
- **Recall**: 0.5356  
- **F1-Score**: 0.5020  

### Per-Class Performance
- **Down**: Precision = 0.4643, Recall = 0.2484  
- **Up**: Precision = 0.5581, Recall = 0.7680  

---

## Key Findings
- The model achieves modest performance with slight class imbalance bias  
- Strong bias toward predicting "Up" orientation  
- Training shows instability with high variance in metrics  
- Subtle visual differences between orientations make classification challenging  

---

## Dependencies
See `requirements.txt` for the complete list. Main packages:
- torch
- torchvision
- scikit-learn
- matplotlib
- Pillow
- jupyter

---

## Troubleshooting

### CUDA out of memory error:
- Reduce batch size (try 16 or 8)

### Images not loading:
- Verify `base_dir` path is correct  
- Ensure dataset folders contain image files  

### Slow training:
- If GPU is not available, consider using Google Colab with GPU support  

---

## Future Improvements
- Apply class weighting to address imbalance  
- Try deeper architectures (ResNet50, EfficientNet)  
- Collect more balanced training data  
- Implement data balancing techniques (oversampling/undersampling)  
- Experiment with different augmentation strategies  
- Use ensemble methods combining multiple models  

---

## Notes
- Image normalization uses ImageNet pre-training statistics  
- Data is split 80–20 for training and test (as provided by Kaggle dataset)  
- Transfer learning leverages features learned from ImageNet  
- Training time: ~5–10 minutes on GPU, ~30+ minutes on CPU  

---

## Author
**Shiva Prasad**

---

## License
Dataset sourced from Kaggle — refer to dataset license for usage rights