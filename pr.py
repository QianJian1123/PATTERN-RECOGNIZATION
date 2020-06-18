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


TRAIN_PATH='./train_data.mat'
TEST_PATH='./test_data.mat'

# define model


# model = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=1) 
# model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None, 
#     min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
#     max_features=None, random_state=None, max_leaf_nodes=None, 
#     min_impurity_decrease=0.0, min_impurity_split=None,
#     class_weight=None)
# model=AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2,min_samples_split=3, min_samples_leaf=1, ),
                        #  algorithm="SAMME",
                        #  n_estimators=100, learning_rate=1)
model=RandomForestClassifier (n_estimators=200, criterion='entropy', max_depth=None,
min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0)

# PREPARE Training Data
raw_train_data=loadmat(TRAIN_PATH)['yidali_train'][0]
train_label=[]
train_data=[]
for num,i in enumerate(raw_train_data):
    # print(i.shape)
    i=i.transpose()
    if num==0:
        i=i[0:-1:2]
    train_data.append(i)
    for j in i:
        train_label.append(num)
# print(len(train_label))
train_label=numpy.array(train_label)
train_data=numpy.concatenate(train_data, axis=0)
print(train_data.shape)

# prepare testing data
raw_test_data=loadmat(TEST_PATH)['yidali_test'][0]
test_label=[]
test_data=[]
for num,i in enumerate(raw_test_data):
    # print(i.shape)
    i=i.transpose()
    test_data.append(i)
    for j in i:
        test_label.append(num)
# print(len(test_label))
test_label=numpy.array(test_label)
test_data=numpy.concatenate(test_data, axis=0)
print(test_data.shape)

# preprocessing dataset
scaler = preprocessing.StandardScaler().fit(train_data)
train_data=scaler.transform(train_data)
test_data=scaler.transform(test_data)
# train_data = preprocessing.normalize(train_data, norm='l2')
# test_data = preprocessing.normalize(test_data, norm='l2')
# calc time 
start_time=time.time()
model.fit(train_data,train_label)
end_time=time.time()
time_cost=end_time-start_time
print('time cost : {}'.format(time_cost))

# test
result=model.score(train_data,train_label)
print(result)
result=model.score(test_data,test_label)
print(result)
result=kappa(model.predict(test_data),test_label)
print('kappa:',result)
result=classification_report(model.predict(test_data),test_label)
print(result)
result=confusion_matrix(model.predict(test_data),test_label)
print('matrix:\n',result)
# save model
with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)
# print(model.get_params())

