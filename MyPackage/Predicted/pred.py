from sklearn.metrics import r2_score, mean_squared_error

from MyPackage.Train_model.train_svm import *
import warnings
warnings.filterwarnings("ignore")


def predicted(label_encoder, X_train, X_test, y_train, y_test, model):
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='micro', zero_division=1)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='micro', zero_division=1)
    pre = precision_score(y_test, y_pred, average='micro', zero_division=1)
    cm = np.array(confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]))

    confusion = pd.DataFrame(cm, index=['is COPD', 'is HC', 'is Asthma', 'is Infected'],
                             columns=['pred COPD', 'pred HC', 'pred Asthma', 'pred Infected'])
    print("Confusion Matrix:\n",confusion)
    print(classification_report(y_test, y_pred))

    data = pd.read_csv('Data/DataFinal.csv')
    des = data.describe(include='all')
    print("Describe:\n",des)

    # Hiển thị kết quả và đánh giá mô hình
    print("Độ chính xác: {:.2f}%".format(accuracy * 100))
    print("Nhớ lại: {:.2f}%".format(recall * 100))
    print("Chính xác dự đoán: {:.2f}%".format(pre * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))
    a = []

    for i in X_test:
        predicted_label = model.predict([i])
        decoded_label = label_encoder.inverse_transform(predicted_label)
        # print("Nhãn dự đoán:", decoded_label)
        a.append(decoded_label)
    return a

def danh_gia_mo_hinh(X_train, X_test, y_train, y_test, model):
    y_pred = model.predict(X_test)  # dự báo y_pred dựa trên tập x_test
    y_pred_train = model.predict(X_train)

    # Đánh giá mô hình bằng một số các metric
    print('--------Kết quả trên dữ liệu huấn luyện-------')
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    print("Mean Squared Error (MSE):", mse_train)
    print("R-squared (R2) Score:", r2_train)

    print('--------Kết quả thẩm định-------')
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2) Score:", r2)
    df_result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_compare = df_result.head(10)
    print(df_compare)
    df_compare.plot(kind='line')
    plt.show()
