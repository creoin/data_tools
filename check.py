# Check the data manager works as expected
from datamanager import *

# Prepare Dataset
# Iris Dataset
# filepath = 'data/iris/iris.data'
# data_manager = IrisData(filepath, (0.7,0.15,0.15))

# Task Dataset
filepath = 'data/task/task1.csv'
data_manager = TaskData(filepath, (0.8,0.10,0.10))


data_manager.init_dataset()
X, Y = data_manager.prepare_train()
X_valid, Y_valid = data_manager.prepare_valid()

print('Train')
print(X[:3])
print(Y[:3])
print(type(X))

print('\n\nValid')
print(X_valid[:3])
print(Y_valid[:3])
print(type(X_valid))


# NLP Spooky Author Identification Dataset
# filepath = 'data/spooky_author_identification/train.csv'
# data_manager = SpookyData(filepath, (0.8, 0.1, 0.1), one_hot_encode=False, output_numpy=False)
# data_manager.init_dataset()
# train_x, train_y = data_manager.prepare_train()
#
# print('Train')
# print(train_x[:3])
# print(train_y[:3])
# print(type(train_x))
