from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

# 假设 feature1 是 38 维的特征，feature2 是 768 维的特征，y 是标签
# feature1, feature2, y = ...
feature1 = pd.read_excel('demo.xlsx')
feature2 = pd.read_excel('bert_vector/bert_vec.xlsx')
feature2.columns = feature2.columns.astype('str')
print(feature2)
print(feature1)
y = pd.read_excel('clean_cut.xlsx')
y = y['label']
# 特征预处理
scaler = StandardScaler()
feature1 = scaler.fit_transform(feature1)
feature2 = scaler.fit_transform(feature2)

print(feature2)
print(feature1)
# 使用单一类别的特征进行训练和测试
for feature in [feature1, feature2]:
    X_train, X_test, y_train, y_test = train_test_split(feature, y, test_size=0.2, random_state=42)
    clf = LinearSVC(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("F1 Score: ", f1_score(y_test, y_pred, average='weighted'))

# 特征融合
features_combined = np.hstack((feature1, feature2))
X_train, X_test, y_train, y_test = train_test_split(features_combined, y, test_size=0.2, random_state=42)
clf = LinearSVC(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy After Feature Fusion: ", accuracy_score(y_test, y_pred))
print("F1 Score After Feature Fusion: ", f1_score(y_test, y_pred, average='weighted'))