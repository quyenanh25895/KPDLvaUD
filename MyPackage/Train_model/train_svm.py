from MyPackage.DataView.Data_view import *

def train_svm(data, data_finally):
    # Label encoding nhãn, split 2 dòng đầu không có giá trị
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['Diagnosis'])[2:]

    # Chia dữ liệu thành bộ train và bộ test (test_size là phần trăm của bộ test)
    X_train, X_test, y_train, y_test = train_test_split(data_finally, y, test_size=0.1, shuffle=True, random_state=6)

    svm_model = SVC(kernel='rbf', C=1000)
    svm_model.fit(X_train, y_train)
    joblib.dump(svm_model, 'Data/models/svm_model_1.joblib')
    return label_encoder, X_train, X_test, y_train, y_test, svm_model
