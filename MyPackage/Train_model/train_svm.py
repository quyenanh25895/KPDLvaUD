import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def train_svm():
    # Đọc file Excel
    data = pd.read_csv('Data/Exasens.csv')

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

    df_t = np.array(df_filled)

    # Chuyển kiểu dữ liệu thành float để nhận data cuối cùng

    data_finally = df_t.astype(float)
    dfinal = pd.DataFrame(data_finally)
    dfinal.to_csv('Data/DataFinal.csv', index=False, header=False)

    # Label encoding nhãn, split 2 dòng đầu không có giá trị
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['Diagnosis'])[2:]

    # Chia dữ liệu thành bộ train và bộ test (test_size là phần trăm của bộ test)
    X_train, X_test, y_train, y_test = train_test_split(data_finally, y, test_size=0.1, shuffle=True, random_state=15)

    svm_model = SVC(kernel='rbf', C=1000)
    svm_model.fit(X_train, y_train)
    joblib.dump(svm_model, 'Data/models/svm_model_1.joblib')
    return label_encoder, X_train, X_test, y_train, y_test, svm_model
