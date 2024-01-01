from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from MyPackage.Train_model.train_svm import *
import matplotlib.pyplot as plt


def kf():
    label_encoder,X_train, X_test, y_train, y_test, model = train_svm()
    kfold = KFold(n_splits=10, shuffle=True, random_state=96)
    acc, pre, f1, rec = [], [], [], []
    label_pred = []

    for i, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        a = []
        svm = SVC(kernel="linear", C=100)
        svm.fit(X_train[train_idx], y_train[train_idx])
        y_pred = svm.predict(X_train[val_idx])

        for j in X_train[val_idx]:
            predicted_label = svm.predict([j])
            decoded_label = label_encoder.inverse_transform(predicted_label)

            a.append(decoded_label)
        label_pred.append(a)
        p, r, f, s = precision_recall_fscore_support(y_true=y_train[val_idx], y_pred=y_pred, zero_division=1,
                                                     average='macro')
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
