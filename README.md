# 🎤 Voice-based Gender Recognition System

This project implements a **deep learning-based system** for recognizing gender from voice recordings. It extracts **Mel-Frequency Cepstral Coefficients (MFCCs)** as features and employs a **Long Short-Term Memory (LSTM) neural network** to classify voices as male or female.

---

## 🚀 Project Overview

The system utilizes the **BVC Gender & Age from Voice dataset**, which includes:

- **Total voice samples:** 3,964
- **Male voice samples:** 2,149
- **Female voice samples:** 1,815
- **Unique speakers:** 560 (363 males, 197 females)

Each audio sample is labeled and preprocessed before being used for training and testing.

---

## 🔧 Key Features

### 1️⃣ Data Preprocessing

- Matched audio files with metadata to ensure correct labeling.
- Converted gender labels into categorical values: **Male (0), Female (1)**.
- Extracted **MFCC features** from each voice sample.
- Split data into **70% training** and **30% testing**.

### 2️⃣ Model Architecture

- **LSTM-based neural network** for sequential voice processing.
- **Model Layers:**
  - LSTM layer (128 units) for feature extraction.
  - Dense layers (64, 32, 16 units) with ReLU activation.
  - Dropout layers (0.2) to prevent overfitting.
  - Softmax output (2 units) for binary classification.
- Optimized using **Adam optimizer** with a learning rate of 0.0002.

### 3️⃣ Model Training & Evaluation

- **Trained for:** 70 epochs with batch size = 512.
- **Loss function:** Categorical Cross-Entropy.
- **Evaluation Metrics:** Accuracy and Loss.
- Achieved **79.33% accuracy** on the test dataset.

---

## 📊 Results & Insights

- Training accuracy improved steadily, reaching **77%**.
- Validation accuracy remained stable, indicating **good generalization**.
- Model predictions on unseen samples demonstrated **reliable gender classification**.

---

## 🖥️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/AbdelkadirSellahi/voice-gender-recognition.git
cd voice-gender-recognition
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Jupyter Notebook

```bash
jupyter notebook Voice-based Gender Recognition System.ipynb
```

### 4️⃣ Predict Gender from a Sample Audio File

- Load a new audio file and extract **MFCC features**.
- Use the trained model to predict gender.

---

## 📄 Conclusion

This project successfully implements a **voice-based gender recognition** system using deep learning. The **LSTM model** effectively learns speech patterns and classifies gender with high accuracy.

### 🔮 Future Improvements:

- Expanding the dataset for better gender balance.
- Experimenting with **transformer-based** models.
- Optimizing the model for **real-time applications**.

---

## 📚 Required Libraries

- **Librosa**: Audio analysis and feature extraction.
- **Keras**: Deep learning framework for model development.
- **Scikit-learn**: Data preprocessing and evaluation metrics.

---

## 💬 **Contact**

Feel free to open an issue or reach out for collaboration!  

**Author**: *Abdelkadir Sellahi*

**Email**: *abdelkadirsellahi@gmail.com* 

**GitHub**: [Abdelkadir Sellahi](https://github.com/AbdelkadirSellahi)
---

## 🙏 Acknowledgments

- **Dataset:** [BVC Gender & Age from Voice](https://www.kaggle.com/datasets/ogechukwu/voice)
- **Libraries:** Keras, Librosa, Scikit-learn

