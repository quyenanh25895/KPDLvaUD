from MyPackage.Train_model.train_svm import *


def predicted(label_encoder, X_train, X_test, y_train, y_test, model):
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='micro')
    pre = precision_score(y_test, y_pred, average='micro')
    cm = np.array(confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]))

    confusion = pd.DataFrame(cm, index=['is COPD', 'is HC', 'is Asthma', 'is Infected'],
                             columns=['pred COPD', 'pred HC', 'pred Asthma', 'pred Infected'])
    print(confusion)
    print(classification_report(y_test, y_pred))

    data = pd.read_csv('Data/DataFinal.csv')
    des = data.describe(include='all')
    print(des)

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