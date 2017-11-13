# Check the data manager works as expected
from datamanager import *

# Prepare Dataset
# Iris Dataset
filepath = 'data/iris/iris.data'
data_manager = IrisData(filepath, (0.7,0.15,0.15))

data_manager.init_dataset()
X, Y = data_manager.prepare_train()
X_valid, Y_valid = data_manager.prepare_valid()

print('Train')
print(X[:3])
print(Y[:3])

print('\n\nValid')
print(X_valid[:3])
print(Y_valid[:3])
