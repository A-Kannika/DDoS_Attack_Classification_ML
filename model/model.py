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

def read_data():
    df = pd.read_csv("../data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    print(df.head())
    return df

def preprocess_data(df):
    # prepare the data
    # 1. remove the space before the column names
    df.columns = df.columns.str.strip()
    # 2. Check the unique values in the label target columns
    print(df.loc[:, 'Label'].unique())



def main():
    df = read_data()
    preprocess_data(df)

if __name__ == "__main__":
    main()