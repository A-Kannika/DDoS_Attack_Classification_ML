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
    plt.figure(figsize=(10,5))
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
    plt.show()
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
    return df


def main():
    df = read_data()
    preprocess_data(df)
    checking_null(df)
    plotMissingValues(df)
    df = remove_null(df)
    df = change_datatype(df)
    # print(df.describe())
    # hist_for_each_feature(df)
    df = data_splitting(df)
    

if __name__ == "__main__":
    main()