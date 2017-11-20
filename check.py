# Check the data manager works as expected
from datamanager import *

# Prepare Dataset
# Iris Dataset
# filepath = 'data/iris/iris.data'
# data_manager = IrisData(filepath, (0.7,0.15,0.15))

# Task Dataset
# filepath = 'data/task/task1.csv'
# data_manager = TaskData(filepath, (0.8,0.10,0.10))
#
#
# data_manager.init_dataset()
# X, Y = data_manager.prepare_train()
# X_valid, Y_valid = data_manager.prepare_valid()
#
# print('Train')
# print(X[:3])
# print(Y[:3])
# print(type(X))
#
# print('\n\nValid')
# print(X_valid[:3])
# print(Y_valid[:3])
# print(type(X_valid))


from vocabulary import *

# NLP Spooky Author Identification Dataset
filepath = 'data/spooky_author_identification/train.csv'
data_manager = SpookyData(filepath, (0.8, 0.1, 0.1), one_hot_encode=False, output_numpy=False)
data_manager.init_dataset()
train_x, train_y = data_manager.prepare_train()

# Vocabulary
vocab = Vocabulary('data/spooky_author_identification/tmp', 20000)
vocab.build_vocabulary(train_x, train_y)

sents_vocab, rev_sents_vocab = vocab.get_sentence_vocabulary()
label_vocab, rev_label_vocab = vocab.get_label_vocabulary()

train_x_tok = vocab.data_to_token_ids(train_x, 'train')
train_y_tok = vocab.labels_to_token_ids(train_y, 'train')

train_set = list(zip(train_x_tok, train_y_tok))

vocab.translate_examples(train_set[:5])

print('Train')
print(train_x[:3])
print(train_y[:3])
print(type(train_x))

from batches import *
print('\n\n\n')
print('Check Batches')
print('Make 3 epochs of batches of 2 elements, from 6 examples')

batches = Batches(2)
for i, epoch in enumerate(batches.gen_padded_batch_epochs(train_set[:6], 3)):
    print('\n\nEpoch {}'.format(i))
    for step, (batch_x, batch_y, lengths) in enumerate(epoch):
        print('{:5d}\n{}\n{}\n\n'.format(step, batch_x, batch_y))
