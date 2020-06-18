from scipy.io import loadmat 
import numpy,time
from sklearn import tree 
import pickle
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
x=[1,2,3,4,5,6,7,8,9]
Y1=[0.2,0.94,0.9,0.91,0.79,0.83,0.98,1.00,0.97]
Y2=[0.63,0.87,
    0.71, 
    0.86 ,
    0.78 ,
    0.85 ,
    0.96 ,
    0.98 ,
    0.67 ]
Y3=[
    0.97,
    0.33,
    0.14,
    0.37,
    0.36,
    0.67,
    0.56,
    0.89,
    0.16
]
plt.figure()
plt.title('Result Analysis')
plt.plot(x, Y1, color='green', label='Sub_R=100 accuracy')
plt.plot(x, Y2, color='red', label='Sub_R=10 accuracy')
plt.plot(x, Y3, color='blue', label='Sub_R=1 accuracy')
plt.show() 
# TRAIN_PATH='./train_data.mat'
# TEST_PATH='./test_data.mat'



# # model = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=1) 
# model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None, 
#     min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
#     max_features=None, random_state=None, max_leaf_nodes=None, 
#     min_impurity_decrease=0.0, min_impurity_split=None,
#     class_weight=None)
# # model=AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2,min_samples_split=3, min_samples_leaf=1, ),
#                         #  algorithm="SAMME",
#                         #  n_estimators=100, learning_rate=1)
# model=RandomForestClassifier (n_estimators=200, criterion='entropy', max_depth=None,
# min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0)

# # PREPARE Training Data
# raw_train_data=loadmat(TRAIN_PATH)['yidali_train'][0]
# x=[1,2,3,4,5,6,7,8,9]
# y=[]
# plt.figure()
# for num,i in enumerate(raw_train_data):
#     y.append(len(i[0]))
# plt.plot(x,y)
# plt.show()  
# train_label=[]
# train_data=[]
# for num,i in enumerate(raw_train_data):
#     # print(i.shape)
#     i=i.transpose()
#     if num==0:
#         i=i[0:-1:2]
#     train_data.append(i)
#     for j in i:
#         train_label.append(num)
# # print(len(train_label))
# # plt.figure()
# # plt.plot(train_data[1][1])
# # plt.show()
# train_label=numpy.array(train_label)
# plt.figure()
# scaler = preprocessing.StandardScaler().fit(numpy.concatenate(train_data, axis=0))

# for i,train_datai in enumerate(train_data):
#     plt.subplot(331+i)
#     plt.title('Class : {}'.format(i+1))
#     train_datai=scaler.transform(train_datai)
#     plt.plot(train_datai[1])
# plt.show()  
# train_data=numpy.concatenate(train_data, axis=0)
# print(train_data.shape)

# # prepare testing data
# raw_test_data=loadmat(TEST_PATH)['yidali_test'][0]
# test_label=[]
# test_data=[]
# for num,i in enumerate(raw_test_data):
#     # print(i.shape)
#     i=i.transpose()
#     test_data.append(i)
#     for j in i:
#         test_label.append(num)
# # print(len(test_label))
# test_label=numpy.array(test_label)
# test_data=numpy.concatenate(test_data, axis=0)
# print(test_data.shape)
# # plt.figure()
# # plt.plot(train_data[1])
# # plt.show()  
# # preprocessing dataset
# # scaler = preprocessing.StandardScaler().fit(train_data)
# # train_data=scaler.transform(train_data)
# # test_data=scaler.transform(test_data)
# # train_data = preprocessing.normalize(train_data, norm='l2')
# # test_data = preprocessing.normalize(test_data, norm='l2')
# # plt.figure()
# # plt.plot(train_data[1])
# # plt.show()  
74895  1435  1055   640   280   105  2225    42  1619
504     0   456     0     0     0     4     0     0
893    12     0  1165   134     0     0     0     0
319     2     0    36   632     0     0     0     0
218    30     0     0     0   530     0     0     0
201     0     0     0     0     0     0   631     0
475     0     0     0     0     0    16     0   815