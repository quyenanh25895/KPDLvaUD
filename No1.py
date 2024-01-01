import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Đọc file Excel
data = pd.read_csv('Data/Exasens.csv')

# Tách dữ liệu đầu vào (features) và nhãn (labels)
X = data.drop('Diagnosis', axis=1)
X = np.array(X)

# Dữ liệu training chưa khử NaN bỏ id và nhãn
data_init = X[2:, 1:8]
n = data_init.shape[0]
m = data_init.shape[1]
df = pd.DataFrame(data_init)

# Tạo một đối tượng SimpleImputer với chiến lược điền giá trị trung bình
#most_frequent median
imputer = SimpleImputer(strategy='mean')

# Điền giá trị trung bình vào các giá trị khuyết trong DataFrame
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

df_t = np.array(df_filled)

# Chuyển kiểu dữ liệu thành float để nhận data cuối cùng
df_file = pd.DataFrame(df_t)
df_file.to_csv('Data/DataFinal.csv', index=False)
data_finally = df_t.astype(float)

# Label encoding nhãn, split 2 dòng đầu không có giá trị
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Diagnosis'])[2:]

# Chia dữ liệu thành bộ train và bộ test (test_size là phần trăm của bộ test)
X_train, X_test, y_train, y_test = train_test_split(data_finally, y, test_size=0.1, shuffle=True)

c = int(input('Nhap tham so C cua mo hinh, C = '))
svm_model = SVC(kernel='linear', C=1000)
svm_model.fit(X_train, y_train)
yh_pred = svm_model.predict(X_test)

h_accuracy = accuracy_score(y_test, yh_pred)
print("Độ chính xác trên bộ test:", h_accuracy)
a = []

for i in X_test:
    predicted_label = svm_model.predict([i])
    decoded_label = label_encoder.inverse_transform(predicted_label)
    print("Nhãn dự đoán:", decoded_label)
    a.append(decoded_label)

kfold = KFold(n_splits=10, shuffle=True, random_state=96)
acc, pre, f1, rec = [], [], [], []
label_pred = []

for i, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    a = []
    svm = SVC(kernel="linear", C=c)
    svm.fit(X_train[train_idx], y_train[train_idx])
    y_pred = svm.predict(X_train[val_idx])

    for j in X_train[val_idx]:
        predicted_label = svm.predict([j])
        decoded_label = label_encoder.inverse_transform(predicted_label)
        # print("Nhãn dự đoán:", decoded_label)
        a.append(decoded_label)
    label_pred.append(a)
    p, r, f, s = precision_recall_fscore_support(y_true=y_train[val_idx], y_pred=y_pred, zero_division=1, average='macro')
    acc.append(accuracy_score(y_train[val_idx], y_pred))
    pre.append(p)
    rec.append(r)
    f1.append(f)

plt.plot(acc, marker='.')
plt.plot(pre, marker='.')
plt.plot(rec, marker='.')
plt.plot(f1, marker='.')
plt.legend(['Acc', 'Pre', 'Rec', 'F1'])
plt.show()
d = pd.DataFrame(label_pred)
d.to_csv('Final/Predicted.csv', index=True)
print('Accuracy:\n ', acc)
print('Precision:\n', pre)
print('Recall:\n', rec)
print('F1:\n', f1)