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

    plt.figure(figsize=(4,4))
    plt.bar([0, 1], counts, color=['green', 'red'])
    plt.xticks([0, 1], labels=['Not Null', 'Has Null'])
    plt.ylabel('Number of Columns')
    plt.title('Columns with Null Values')
    plt.show()

def plotMissingValues(dataframe):
    missing_values = dataframe.isnull().sum()  # Counting null values for each column
    fig = plt.figure(figsize=(16, 6))
    missing_values.plot(kind='bar')
    plt.xlabel("Features")
    plt.ylabel("Missing values")
    plt.title("Total number of Missing values in each feature")
    plt.show()

def main():
    df = read_data()
    preprocess_data(df)
    checking_null(df)
    plotMissingValues(df)

if __name__ == "__main__":
    main()