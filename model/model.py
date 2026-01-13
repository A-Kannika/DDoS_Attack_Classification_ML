import pandas as pd
import numpy as np

# For ploting the graphs
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine learning Model 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Machine learning model evaluation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.tree import plot_tree

def read_data():
    df = pd.read_csv("../data/DDos.csv")
    # df = pd.read_csv("../data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    # df = pd.read_csv("../data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    # df = pd.read_csv("../data/Friday-WorkingHours-Morning.pcap_ISCX.csv")
    # df = pd.read_csv("../data/Monday-WorkingHours.pcap_ISCX_part_1.csv")
    # df = pd.read_csv("../data/Monday-WorkingHours.pcap_ISCX_part_2.csv")
    # df = pd.read_csv("../data/Monday-WorkingHours.pcap_ISCX_part_3.csv")
    # df = pd.read_csv("../data/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
    # df = pd.read_csv("../data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    # df = pd.read_csv("../data/Tuesday-WorkingHours.pcap_ISCX_part_1.csv")
    # df = pd.read_csv("../data/Tuesday-WorkingHours.pcap_ISCX_part_2.csv")
    # df = pd.read_csv("../data/Tuesday-WorkingHours.pcap_ISCX_part_3.csv")
    # df = pd.read_csv("../data/Wednesday-workingHours.pcap_ISCX_part_1.csv")
    # df = pd.read_csv("../data/Wednesday-workingHours.pcap_ISCX_part_2.csv")
    # df = pd.read_csv("../data/Wednesday-workingHours.pcap_ISCX_part_3.csv")
    
    print(df.head())
    print(df.info(show_counts=True))
    return df

def preprocess_data(df):
    # prepare the data
    # 1. remove the space before the column names
    df.columns = df.columns.str.strip()
    # 2. Check the unique values in the label target columns
    print(df.loc[:, 'Label'].unique())

def checking_null(df):
    # Count columns with nulls and without nulls
    num_null = (df.isna().sum() > 0).sum()       # columns that have at least 1 null
    num_not_null = df.shape[1] - num_null        # columns with no nulls
    # Data for histogram
    counts = [num_not_null, num_null]
    plt.figure(figsize=(10,5))
    plt.bar([0, 1], counts, color=['green', 'red'])
    plt.xticks([0, 1], labels=['Not Null', 'Has Null'])
    plt.ylabel('Number of Columns')
    plt.title('Columns with Null Values')
    # plt.show()

def plotMissingValues(dataframe):
    missing_values = dataframe.isnull().sum()  # Counting null values for each column
    fig = plt.figure(figsize=(16, 6))
    missing_values.plot(kind='bar')
    plt.xlabel("Features")
    plt.ylabel("Missing values")
    plt.title("Total number of Missing values in each feature")
    # plt.show()

def remove_null(df):
    data_f = df.dropna()
    #  plotMissingValues(data_f) # you should not see any null data in the graph
    return data_f

def change_datatype(df):
    # map the data to numerical data
    df['Label'] = df['Label'].map({'BENIGN': 0, 'DDoS': 1})

    counts = df['Label'].value_counts().sort_index()

    plt.figure(figsize=(10,5))
    plt.bar(counts.index, counts.values, color=['green', 'red'])
    plt.xticks([0, 1], labels=['BENIGN', 'DDoS'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Label Distribution')
    # plt.show()
    return df

def hist_for_each_feature(df):
    plt.figure(5)
    for col in df.columns:
        plt.hist(df[col])
        plt.title(col)
        plt.show()

def data_splitting(df):
    #  split data into features and target variable
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    print("The train dataset size = ",X_train.shape)
    print("The test dataset size = ",X_test.shape)
    return X_train, X_test, y_train, y_test

def random_forest_model(X_train, X_test, y_train, y_test):
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    #  Getting feature importances from the trained model
    importances = rf_model.feature_importances_

    # getting the indices of features sorted by importance
    indices = sorted(range(len(importances)), key = lambda i:importances[i], reverse=False)
    # Replace with your column names
    feature_names = [f"Features {i}" for i in indices]

    # Plotting feature importances horizontally
    plt.figure(figsize=(8, 14))
    plt.barh(range(X_train.shape[1]), importances[indices], align="center")
    plt.yticks(range(X_train.shape[1]), feature_names)
    plt.xlabel("Importance")
    plt.title("Feature Importances")
    plt.show()

    # tree_plot(rf_model)

    return rf_model, rf_pred

def logistic_regression_model(X_train, X_test, y_train, y_test):
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    return lr_model, lr_pred

def neural_network_model(X_train, X_test, y_train, y_test):
    nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10, random_state=42)
    nn_model.fit(X_train, y_train)
    nn_pred = nn_model.predict(X_test)

    return nn_model, nn_pred

def tree_plot(rf_model):
    estimator = rf_model.estimators_[0]  # Selecting the first estimator from the random forest model
    plt.figure(figsize=(20, 10))
    plot_tree(estimator, filled=True, rounded=True)
    plt.show()

# Function to generate and display a detailed confusion matrix
def plot_confusion_matrix(y_test, y_pred, classes, title):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def random_forest_model_evaluation(y_test, y_pred):
    # Evaluate Random Forest
    rf_accuracy = accuracy_score(y_test, y_pred)
    rf_f1 = f1_score(y_test, y_pred)
    rf_precision = precision_score(y_test, y_pred)
    rf_recall = recall_score(y_test, y_pred)

    print('\nRandom Forest Metrics:')
    print(f'Accuracy: {rf_accuracy:.4f}')
    print(f'F1 Score: {rf_f1:.4f}')
    print(f'Precision: {rf_precision:.4f}')
    print(f'Recall: {rf_recall:.4f}')
    plot_confusion_matrix(y_test, y_pred, ['Benign', 'DDoS'], 'Random Forest Confusion Matrix')

def logistic_regression_model_evaluation(y_test, y_pred):
    lr_accuracy = accuracy_score(y_test, y_pred)
    lr_f1 = f1_score(y_test, y_pred)
    lr_precision = precision_score(y_test, y_pred)
    lr_recall = recall_score(y_test, y_pred)
    print('\nLogistic Regression Metrics:')
    print(f'Accuracy: {lr_accuracy:.4f}')
    print(f'F1 Score: {lr_f1:.4f}')
    print(f'Precision: {lr_precision:.4f}')
    print(f'Recall: {lr_recall:.4f}')
    plot_confusion_matrix(y_test, y_pred, ['Benign', 'DDoS'], 'Logistic Regression Confusion Matrix')

def neural_network_model_evaluation(y_test, y_pred):
    nn_accuracy = accuracy_score(y_test, y_pred)
    nn_f1 = f1_score(y_test, y_pred)
    nn_precision = precision_score(y_test, y_pred)
    nn_recall = recall_score(y_test, y_pred)

    print('\nNeural Network Metrics:')
    print(f'Accuracy: {nn_accuracy:.4f}')
    print(f'F1 Score: {nn_f1:.4f}')
    print(f'Precision: {nn_precision:.4f}')
    print(f'Recall: {nn_recall:.4f}')
    plot_confusion_matrix(y_test, y_pred, ['Benign', 'DDoS'], 'Logistic Regression Confusion Matrix')

def model_comparison(X_test, y_test, rf_model, lr_model, nn_model):
    # Random Forest
    rf_proba = rf_model.predict_proba(X_test)

    # Logistic Regression
    lr_proba = lr_model.predict_proba(X_test)

    # Neural Network
    nn_proba = nn_model.predict_proba(X_test)

    # Combine predictions for ROC curve
    # Calculate ROC curve for Random Forest
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba[:, 1])
    rf_auc = auc(rf_fpr, rf_tpr)

    # Calculate ROC curve for Logistic Regression
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba[:, 1])
    lr_auc = auc(lr_fpr, lr_tpr)

    # Calculate ROC curve for Neural Network
    nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_proba[:, 1])
    nn_auc = auc(nn_fpr, nn_tpr)

    # Plot ROC curves for all models
    plt.figure(figsize=(8, 6))
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
    plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
    plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.2f})')

    # Plot ROC curve for random classifier (50% area)
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Classifier (AUC = 0.50)')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    df = read_data()
    preprocess_data(df)
    checking_null(df)
    #  plotMissingValues(df)
    df = remove_null(df)
    df = change_datatype(df)
    # print(df.describe())
    # hist_for_each_feature(df)
    X_train, X_test, y_train, y_test = data_splitting(df)

    rf_model, rf_pred = random_forest_model(X_train, X_test, y_train, y_test)
    random_forest_model_evaluation(y_test, rf_pred)

    lr_model, lr_pred = logistic_regression_model(X_train, X_test, y_train, y_test)
    logistic_regression_model_evaluation(y_test, lr_pred)

    nn_model, nn_pred = neural_network_model(X_train, X_test, y_train, y_test)
    neural_network_model_evaluation(y_test, nn_pred)

    model_comparison(X_test, y_test, rf_model, lr_model, nn_model)



    

if __name__ == "__main__":
    main()