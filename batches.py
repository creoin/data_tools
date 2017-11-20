"""
Class for arranginge padded batches.
"""
import copy
from random import shuffle

class Batches(object):
    def __init__(self, batch_size, pad_sym=0):
        self.batch_size = batch_size
        self.pad_sym = pad_sym

    def pad_batch(self, batch):
        max_len = max([len(b_seq) for b_seq in batch])
        lengths = []

        for idx, batched_seq in enumerate(batch):
            current_len = len(batched_seq)
            lengths.append(current_len)
            if current_len < max_len:
                padding = [self.pad_sym for _ in range(max_len - current_len)]
                batched_seq.extend(padding)
        return batch, lengths

    def gen_padded_batches(self, data):
        X, Y = zip(*data)
        data_len = len(X)
        n_steps = data_len // self.batch_size

        for step in range(n_steps):
            idx_start = self.batch_size * step
            idx_end   = self.batch_size * (step+1)
            batch_x = X[ idx_start : idx_end ]
            batch_y = Y[ idx_start : idx_end ]
            padded_x, lengths = self.pad_batch(batch_x)
            yield (padded_x, batch_y, lengths)

    def gen_padded_batch_epochs(self, data, num_epochs):
        for i in range(num_epochs):
            shuffled_data = copy.deepcopy(data)
            shuffle(shuffled_data)
            yield self.gen_padded_batches(shuffled_data)
