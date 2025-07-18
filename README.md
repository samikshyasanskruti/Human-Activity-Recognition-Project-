#  Human Activity Recognition using CNN-BiLSTM

This project implements a hybrid deep learning model combining **Convolutional Neural Networks (CNN)** and **Bidirectional Long Short-Term Memory (BiLSTM)** to classify human activities based on smartphone sensor data. It uses the well-known **UCI HAR dataset** and achieves high accuracy in detecting daily activities like walking, sitting, standing, and more.

---

##  Overview

- **Domain:** Time-Series Classification  
- **Model:** CNN + BiLSTM  
- **Dataset:** UCI Human Activity Recognition Using Smartphones  
- **Accuracy:** 93.55%  
- **ROC AUC:** 0.9960  
- **Use Case:** Fitness tracking, healthcare monitoring, smart environments

---

##  Dataset Description

The dataset includes smartphone accelerometer and gyroscope data from **30 subjects** performing **6 activities**:

-  Standing  
-  Sitting  
-  Walking  
-  Walking Upstairs  
-  Walking Downstairs  
-  Laying

Each record is a **561-feature vector** derived from the time and frequency domain of raw sensor signals.

 [UCI HAR Dataset Download](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip)

---

##  Technologies Used

| Tool/Library      | Purpose                               |
|-------------------|----------------------------------------|
| Python            | Main language                         |
| NumPy, Pandas     | Data loading, manipulation            |
| Seaborn, Matplotlib | Visualization                        |
| Scikit-learn      | Metrics, preprocessing                 |
| TensorFlow / Keras| Deep Learning (CNN, LSTM)              |

---

##  Model Architecture

```text
Input: (561, 1)
↓
Conv1D (64 filters, ReLU)
↓
MaxPooling1D (pool=2)
↓
Dropout (0.5)
↓
Bidirectional LSTM (64 units)
↓
Flatten
↓
Dense (64 units, ReLU)
↓
Dropout (0.5)
↓
Dense (6 units, Softmax)
```




- **Loss:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Epochs:** 15  
- **Batch Size:** 64  

---

##  Results

| Metric              | Value     |
|---------------------|-----------|
| Test Accuracy        | 93.55%    |
| ROC AUC Score        | 0.9960    |
| F1-Score (Per Class) | > 0.90    |

###  Classification Report Highlights:
- Precision: 0.87 – 1.00  
- Recall: 0.88 – 1.00  

###  Accuracy & Loss Curves:
- Plotted over 15 epochs for training and validation sets.

---

##  Test Case Evaluation

Sample predictions with confidence:

| Test Index | True Activity | Predicted Activity | Confidence | Result |
|------------|----------------|---------------------|------------|--------|
| 0          | WALKING        | WALKING             | 98.42%     |  Pass |
| 53         | STANDING       | STANDING            | 95.76%     |  Pass |
| 250        | SITTING        | SITTING             | 92.31%     |  Pass |
| 120        | SITTING        | STANDING            | 88.67%     |  Fail |
| 600        | LAYING         | LAYING              | 97.92%     |  Pass |

Bar plots are used to visualize the confidence score and correctness.

---

##  Limitations

- Data collected with phone placed only on waist  
- Trained on a small group (30 users)  
- Offline only (not yet optimized for real-time use)  
- Sensitive to unseen environments or sensor placement  

---

##  Future Improvements

- Deploy model in real-time on smartphones  
- Add multiple sensor placements or body parts  
- Integrate with audio, GPS, or video  
- Use transformer or attention-based models  
- Hyperparameter tuning, cross-validation  

---

##  Authors

- **Samikshya Sanskruti Swain** – Reg. No: 2341019634  
- **Kunal Routray** – Reg. No: 2341018202  

 Guided by: **Mr. Gyana Ranjan Patra**  
 Center for AI & ML, CSE Dept.  
Siksha ‘O’ Anusandhan (Deemed to be University), Bhubaneswar

---

##  References

- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)  
- [Keras Documentation](https://keras.io/)  
- [Scikit-learn](https://scikit-learn.org/stable/)  
- [NumPy](https://numpy.org/)  
- [Seaborn](https://seaborn.pydata.org/)

---

##  License

This project is intended for educational and research use only.  
Please cite the UCI HAR dataset if used for publication or deployment.
