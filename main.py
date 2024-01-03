from MyPackage.KFold.kfold import *
from MyPackage.Predicted.pred import *

print("Bat dau huan luyen:")
label_encoder, X_train, X_test, y_train, y_test, model = train_svm()
label = predicted(label_encoder, X_train, X_test, y_train, y_test, model)
print("Hoan tat huan luyen:")
print('KFold:', "_"*30)
kfold_test(label_encoder, X_train, X_test, y_train, y_test)

d_label = pd.DataFrame(label)
d_label.to_csv('Final/Predicted.csv', index=False)

