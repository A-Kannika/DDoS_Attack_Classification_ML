# DDoS Attack Classification Using Machine Learning

## Project Overview
Distributed Denial of Service (DDoS) attacks pose a serious threat to modern networks by overwhelming systems with malicious traffic, leading to service disruption and financial loss. Traditional rule-based detection systems often fail to detect evolving attack patterns. This project explores the use of machine learning techniques to accurately classify DDoS and other network attacks using real-world traffic data.

The goal of this project is to build, train, and evaluate multiple machine learning models for intrusion detection and compare their performance using standard classification metrics.

## Dataset
The dataset used in this project is the CICIDS2017 dataset, published by the Canadian Institute for Cybersecurity (CIC), University of New Brunswick.

### Dataset link:
http://www.unb.ca/cic/datasets/IDS2017.html

### Data Collection Period
- Start: Monday, July 3, 2017 (9:00 a.m.)
- End: Friday, July 7, 2017 (5:00 p.m.)
- Total duration: 5 days

## Data Pre-processing
- Handling missing and infinite values
- Feature scaling and normalization
- Label encoding of attack categories
- Removal of irrelevant or redundant features
- Proper preprocessing ensures model stability and improves learning performance.

## Model Training
Three different machine learning models were trained and evaluated:

### 1. Random Forest Classifier
- Ensemble-based model
- Handles non-linear relationships well
- Robust to noise and overfitting

### 2. Logistic Regression
- Linear classification model
- Strong baseline for binary classification
- Fast and interpretable

### 3. Neural Network
- Multi-layer feedforward neural network
- Capable of learning complex patterns
- Suitable for high-dimensional data

## Model Evaluation Metrics
The following evaluation metrics were used:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- Area Under the Curve (AUC)

## Model-Evaluation Results
| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Random Forest       | 0.9995   | 1.0000    | 0.9990 | 0.9995   |
| Logistic Regression | 0.9441   | 0.9086    | 0.9939 | 0.9494   |
| Neural Network      | 0.9811   | 0.9713    | 0.9935 | 0.9823   |

The high accuracy obtained, particularly by the Random Forest model, is consistent with prior work on CICIDS2017. However, such performance reflects the controlled offline nature of the dataset and may not directly translate to real-time deployment without further validation.

## ROC Curve and AUC Analysis
<img width="800" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/4baa9eb5-f909-488a-a0b2-07be84169661" />

Receiver Operating Characteristic (ROC) curves were plotted to evaluate the trade-off between True Positive Rate (Recall) and False Positive Rate for each classifier. The Area Under the Curve (AUC) was used as a threshold-independent performance metric.

- **Random Forest** achieved an AUC close to 1.0, indicating near-perfect class separability.
- **Neural Network** also demonstrated strong discrimination capability with a high AUC value.
- **Logistic Regression** showed comparatively lower AUC, reflecting its linear decision boundary and reduced ability to model complex attack patterns.

The ROC analysis confirms that ensemble-based and deep learning models outperform linear classifiers for DDoS attack detection in high-dimensional network traffic data.

## Confusion Matrix Analysis

Confusion matrices were generated for all models to analyze classification errors.

- The **Random Forest classifier** produced extremely low false positives and false negatives, indicating reliable detection of attack traffic.
- The **Neural Network model** showed a small number of misclassifications, primarily between benign and low-intensity attack traffic.
- The **Logistic Regression** exhibited a higher false positive rate, which explains its lower precision score.

These results highlight the importance of model selection when minimizing false alarms in intrusion detection systems.

## Conclusion
This project demonstrates that machine learning models can effectively detect DDoS and other network attacks using real-world traffic data.

Key takeaways:
- Random Forest provides the highest accuracy and robustness
- Neural Networks offer strong performance for complex patterns
- Logistic Regression serves as a reliable baseline model
- Proper data preprocessing is critical for achieving high accuracy

Overall, the results highlight the potential of ML-based intrusion detection systems for improving network security.

## Project Highlights

- Real-world cybersecurity dataset (CICIDS2017)
- Multi-model comparison for DDoS detection
- Near-perfect classification performance with Random Forest
- Comprehensive evaluation using ROC, AUC, and confusion matrices
- End-to-end ML pipeline: preprocessing → training → evaluation
