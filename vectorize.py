import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from random import randint
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

comments = pd.read_excel('clean_cut.xlsx')['words']
labels = pd.read_excel('clean_cut.xlsx')['label']
print(comments[:5])
print(labels[:5])

model = Word2Vec(comments, vector_size=100, window=7, sg=1,negative=5, min_count=1, workers=4)

comment_vectors = []
for comment in comments:
    # 获取每个词的向量，然后在axis=0上取平均值
    comment_vector = np.mean([model.wv[word] for word in comment], axis=0)
    comment_vectors.append(comment_vector)

print(comment_vectors[:5])
columns = ['word2vec_{}'.format(i) for i in range(1,101)]
comment_vectors = pd.DataFrame(comment_vectors, columns=columns)
comment_vectors.to_excel('word2vec.xlsx',index=False)

# XGBOOST 分类器设置
# # 设置参数网格
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [4, 6, 8],
#     'learning_rate': [0.1, 0.3, 0.5],
# }
#
#
# # 定义搜索参数的函数
# def search_params(classifier, param_grid, X, y, cv=5):
#     grid_search = GridSearchCV(classifier, param_grid, cv=cv, scoring='accuracy')
#     grid_search.fit(X, y)
#     return grid_search.best_params_
#
# X_train, X_tests, y_train, y_test = train_test_split(comments,labels, test_size=0.2, random_state=1234)
# # 找到最优参数组合
# best_params = search_params(XGBClassifier(random_state=42), param_grid,X_train,y_train)

# # 定义分类器
# classifier = XGBClassifier(
#     n_estimators=100,
#     max_depth=6,
#     learning_rate=0.3,
#     random_state=42,
# )

# word2vec 模型设置
# # 设置随机搜索的参数范围
# vector_size_range = [100, 200, 300]
# window_range = [3, 5, 7]
# min_count_range = [1, 2, 3]
#
# best_score = 0
# best_params = {}
#
# # 进行随机搜索
# for _ in range(20):  # 进行20轮随机搜索
#     vector_size = vector_size_range[randint(0, len(vector_size_range) - 1)]
#     window = window_range[randint(0, len(window_range) - 1)]
#     min_count = min_count_range[randint(0, len(min_count_range) - 1)]
#
#     # 设置交叉验证
#     kf = KFold(n_splits=5)
#     scores = []
#
#     for train_index, test_index in kf.split(comments):
#         comments_train, comments_test = np.array(comments)[train_index], np.array(comments)[test_index]
#         labels_train, labels_test = np.array(labels)[train_index], np.array(labels)[test_index]
#
#         # 使用选择的参数训练Word2Vec模型
#         model = Word2Vec(comments_train, vector_size=vector_size, window=window, min_count=min_count, workers=4)
#
#         # 使用训练好的模型将训练数据转换为向量
#         vectors_train = [np.mean([model.wv[word] for word in comment if word in model.wv], axis=0) for comment in comments_train]
#
#         # 使用训练好的模型将测试数据转换为向量
#         vectors_test = [np.mean([model.wv[word] for word in comment if word in model.wv], axis=0) for comment in comments_test]
#
#         # 使用分类器进行训练和预测
#         classifier.fit(vectors_train, labels_train)
#         predictions = classifier.predict(vectors_test)
#
#         # 计算在测试集上的准确率并保存
#         score = accuracy_score(labels_test, predictions)
#         scores.append(score)
#
#     # 计算平均分数
#     avg_score = np.mean(scores)
#
#     # 如果这个分数比之前的最好分数还要好，那么保留这个模型的参数
#     if avg_score > best_score:
#         best_score = avg_score
#         best_params = {
#             'vector_size': vector_size,
#             'window': window,
#             'min_count': min_count,
#         }
#
# print('Best score:', best_score)
# print('Best params:', best_params)