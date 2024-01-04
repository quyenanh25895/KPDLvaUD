from MyPackage.KFold.kfold import *
from MyPackage.Predicted.pred import *
from MyPackage.DataView.Data_view import *

data = read_data()
data_finally = preprocessing_data(data)
print("Bat dau huan luyen:")

label_encoder, X_train, X_test, y_train, y_test, model = train_svm(data, data_finally)
label = predicted(label_encoder, X_train, X_test, y_train, y_test, model)
print("Hoan tat huan luyen:")
print('KFold:', "_"*30)
kfold_test(label_encoder, X_train, X_test, y_train, y_test)
print("Danh gia mo hinh:")
danh_gia_mo_hinh(X_train, X_test, y_train, y_test, model)
d_label = pd.DataFrame(label)
d_label.to_csv('Final/Predicted.csv', index=False)
