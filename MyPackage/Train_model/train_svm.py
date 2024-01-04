from sklearn.model_selection import GridSearchCV

from MyPackage.DataView.Data_view import *

def train_svm(data, data_finally):
    # Label encoding nhãn, split 2 dòng đầu không có giá trị
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['Diagnosis'])[2:]

    # Chia dữ liệu thành bộ train và bộ test (test_size là phần trăm của bộ test)
    X_train, X_test, y_train, y_test = train_test_split(data_finally, y, test_size=0.1, shuffle=True, random_state=6)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    svm_model = SVC()
    # Sử dụng Grid Search để tìm ra bộ tham số tốt nhất
    grid_search = GridSearchCV(svm_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # In ra bộ tham số tốt nhất
    print("Bộ tham số tốt nhất:", grid_search.best_params_)
    joblib.dump(grid_search, 'Data/models/svm_model_1.joblib')
    return label_encoder, X_train, X_test, y_train, y_test, grid_search
