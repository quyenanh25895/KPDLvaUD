import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
def read_data():
    data = pd.read_csv('Data/Exasens.csv')
    print('-'*30)
    print(data.head(15))
    print('-'*30)
    print(data.info())
    print('-'*30)
    print(data.keys())
    print('-'*30)
    print(data.describe())
    print('-'*30)
    print(data.isnull().sum())
    return data

def preprocessing_data(data):
    # Tách dữ liệu đầu vào (features) và nhãn (labels)
    X = data.drop('Diagnosis', axis=1)
    X = np.array(X)

    # Dữ liệu training chưa khử NaN bỏ id và nhãn
    data_init = X[2:, 1:8]
    df = pd.DataFrame(data_init)

    # Tạo một đối tượng SimpleImputer với chiến lược điền giá trị trung bình
    # most_frequent median
    imputer = SimpleImputer(strategy='mean')

    # Điền giá trị trung bình vào các giá trị khuyết trong DataFrame
    df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


    plt.figure(figsize=(10, 6))
    correlation_matrix = df_filled.corr().round(2)
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True)
    plt.title("Bieu do tuong quan giua cac cot")
    plt.show()
    df_t = np.array(df_filled)
    data_finally = df_t.astype(float)

    #Chuan hoa du lieu ve doan 0 - 1
    scaler = MinMaxScaler()
    data_finally = scaler.fit_transform(data_finally)
    dfinal = pd.DataFrame(data_finally)
    dfinal.to_csv('Data/DataFinal.csv', index=False, header=False)
    print(data_finally)
    return data_finally
