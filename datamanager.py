import os
import numpy as np
import csv
import copy
from random import randint
from random import shuffle

class DataManager(object):
    """
    Class to manage data handling. Just CSV import for now.
    """
    def __init__(self, filepath, split):
        self.filepath = filepath
        self.split = split       # train/valid/test fractions, should sum to 1
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.indexes = set()
        self.initialised = False
        self.discard_header = False

        self.num_classes = 0  # Number of classes in dataset
        self.dataset_path = './data/dataset_name'  # replace dataset_name in child class

    def init_dataset(self):
        self.split_data_paths = {'train': os.path.join(self.dataset_path, 'train.csv'),
                                 'valid': os.path.join(self.dataset_path, 'valid.csv'),
                                 'test' : os.path.join(self.dataset_path, 'test.csv')}
        if os.path.isfile(self.split_data_paths['train']):
            print('Train/Valid/Test data found, loading...')
            self.train_raw, self.valid_raw, self.test_raw = self._load_data()
        else:
            print('Preparing Train/Valid/Test data from {}'.format(self.filepath))
            imported_data = self._import_and_write_data(self.filepath)
            self.train_raw, self.valid_raw, self.test_raw = self._load_from_raw_import(*imported_data)

        # Printouts below to help with DataManager improvements
        # print('\n\nTrain Raw\n')
        # for tr in self.train_raw[:50]:
        #     print(tr)
        # print('num_classes: ', self.num_classes)
        # print('\n\n')

        self.train, self.valid, self.test = self._make_1hot()

        # print('\n\nTrain 1-hot\n')
        # for tr in self.train[:50]:
        #     print(tr)
        # print('num_classes: ', self.num_classes)
        # print('\n\n')

        self.initialised = True
        print('Dataset prepared')

        return True

    def prepare_train(self):
        self._initialise_check()
        return self._prepare_split(self.train)

    def prepare_valid(self):
        self._initialise_check()
        return self._prepare_split(self.valid)

    def prepare_test(self):
        self._initialise_check()
        return self._prepare_split(self.valid)

    def _initialise_check(self):
        if not self.initialised:
            print('Init a dataset first (e.g. data_manager.init_dataset())')
            quit()
        return self.initialised

    def _prepare_split(self, split):
        """
        Separate data split into X and Y and convert to numpy array.
        """
        X, Y = zip(*split)
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    # Loading already split dataset files
    def _load_data(self):
        # self.num_classes = 3
        train_raw = self._load_split(self.split_data_paths['train'])
        valid_raw = self._load_split(self.split_data_paths['valid'])
        test_raw  = self._load_split(self.split_data_paths['test'])
        return train_raw, valid_raw, test_raw

    def _load_split(self, split_path):
        """
        Import data from processed train/valid/test split datasets.
        Override _process_row_split() method in child class for final data structure.
        This should have the final format used in the Neural Network.
        """
        split_raw = []
        with open(split_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                read_line = self._process_row_split(row)
                split_raw.append(read_line)
        return split_raw

    def _load_from_raw_import(self, train_raw, valid_raw, test_raw):
        train_load = self._load_from_raw_import_split(train_raw)
        valid_load = self._load_from_raw_import_split(valid_raw)
        test_load  = self._load_from_raw_import_split(test_raw)
        return train_load, valid_load, test_load

    def _load_from_raw_import_split(self, split):
        """
        Written file, and processed file can have different formats.
        Pass through same processing as if reading from train/valid/test files.
        """
        split_raw = []
        for row in split:
            if not row:
                continue
            read_line = self._process_row_split(row)
            split_raw.append(read_line)
        return split_raw

    # Importing from raw CSV
    def _import_and_write_data(self, filepath):
        self.data_raw = self._data_import(filepath)
        train_raw, valid_raw, test_raw = self._split_data(self.data_raw)
        self._write_split_files(train_raw, valid_raw, test_raw)
        return train_raw, valid_raw, test_raw

    def _data_import(self, filepath):
        """
        Import data from data file.
        Override this _process_row_raw() method in child class for dataset structure.
        Return a list of [property_1, property_2, ..., target_index]
        """
        data_raw = []
        with open(filepath) as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if not row:
                    continue
                if self.discard_header and i == 0:
                    continue
                read_line = self._process_row_raw(row)
                data_raw.append(read_line)
        return data_raw

    def _split_data(self, original_data):
        """
        Split data into training/validation/test sets by the rations in self.split.
        Firstly the data is divided by class, to get an even distribution over all classes.
        """
        train, valid, test = [], [], []
        n_data = len(original_data)
        count = {'train': 0, 'valid': 0, 'test': 0}

        # split by class to fill train/valid/test with even class distribution
        data_by_class = self._split_into_classes(original_data)
        # get the number examples of each class needed for each split
        # e.g. 80% train means 80% of class 1, 80% of class 2, etc.
        n_train, n_valid, n_test = self._get_split_counts_by_class(data_by_class)

        # loop through data one class at a time
        cls_counts = []
        for class_idx, class_data in enumerate(data_by_class):
            cls_count = {'train': 0, 'valid': 0, 'test': 0}
            if not class_data:
                continue
            # pick through class data until empty
            while True:
                pop_idx = randint(0, len(class_data)-1)
                data_line = class_data.pop(pop_idx)
                if cls_count['train'] < n_train[class_idx]:
                    train.append(data_line)
                    cls_count['train'] += 1
                elif cls_count['valid'] < n_valid[class_idx]:
                    valid.append(data_line)
                    cls_count['valid'] += 1
                else:
                    test.append(data_line)
                    cls_count['test'] += 1
                if len(class_data) == 0:
                    break
            # accumulate counts
            for split_name in count.keys():
                count[split_name] += cls_count[split_name]
            cls_counts.append(cls_count)

        for split_set in [train, valid, test]:
            shuffle(split_set)

        self._print_counts(cls_counts, count.keys())

        return train, valid, test

    def _split_into_classes(self, original_data):
        """
        Split data into separate lists by class.
        Last entry of original_data is expected to be the target index, used to split into classes.
        """
        data_by_class = [[] for _ in range(self.num_classes)]
        for data_line in original_data:
            data_by_class[data_line[-1]].append(data_line)  # last entry should be target index
        return copy.deepcopy(data_by_class)

    def _get_split_counts_by_class(self, data_by_class):
        n_splits = []
        n_of_class = [len(class_set) for class_set in data_by_class]
        for fraction in self.split:
            n_split = [int(num * fraction) for num in n_of_class]
            n_splits.append(n_split)
        return n_splits

    def _print_counts(self, cls_counts, split_names):
        # Counts accumulated from processing
        for name in split_names:
            c_counts = [0, 0, 0]
            for c_idx in range(3):
                c_counts[c_idx] = cls_counts[c_idx][name]
            print("Split {} has ({}, {}, {}) examples of each class".format(
                name, c_counts[0], c_counts[1], c_counts[2]))

    # Writing split dataset files
    def _write_split_files(self, train, valid, test):
        self._write_file(train, 'train')
        self._write_file(valid, 'valid')
        self._write_file(test, 'test')

    def _write_file(self, split_data, splitname):
        filename = os.path.join(self.dataset_path, splitname+'.csv')
        with open(filename,'w') as f:
            writer = csv.writer(f, delimiter=',')
            for row in split_data:
                writer.writerow(row)

    def _make_1hot(self):
        train_1hot = self._split_to_1hot(self.train_raw)
        valid_1hot = self._split_to_1hot(self.valid_raw)
        test_1hot  = self._split_to_1hot(self.test_raw)
        return train_1hot, valid_1hot, test_1hot

    def _split_to_1hot(self, split):
        """
        Makes 1-hot encoding from split dataset files.
        Expects last entry of data rows to be the target index
        """
        split_1hot = []
        for data_line in split:
            one_hot = [0 for _ in range(self.num_classes)]
            one_hot[data_line[-1]] = 1
            split_1hot.append([data_line[:-1], one_hot])
        return split_1hot

    def _get_idx(self, label):
        if label in self.label_to_idx:
            return self.label_to_idx[label]
        else:
            new_idx = self.num_classes
            self.label_to_idx[label]   = new_idx
            self.idx_to_label[new_idx] = label
            self.num_classes += 1
            return new_idx

    def _count_idx(self, idx):
        if idx in self.indexes:
            return len(self.indexes)
        else:
            self.indexes.add(idx)
            self.num_classes += 1

    def _process_row_raw(self, row):
        """
        Import lines from raw data file.
        Returns a list of data, the last entry should be a label_id.
        """
        raise NotImplementedError

    def _process_row_split(self, row):
        """
        Import lines from train/valid/test split files.
        Returns a list of data in format used in the Neural Network, the last entry should be a label_id.
        """
        raise NotImplementedError


class IrisData(DataManager):
    def __init__(self, filepath, split):
        super().__init__(filepath, split)
        self.filepath = filepath
        self.split = split       # train/valid/test fractions, should sum to 1
        self.num_classes = 0
        self.dataset_path = './data/iris'

    def _process_row_raw(self, row):
        """
        Import lines from raw Iris data file.
        Imports line of "property_1, property_2, property_3, property_4, label_name"
        Returns list of [property_1, property_2, property_3, property_4, label_index]
        """
        read_line = []
        iris_properties = row[:4]
        iris_properties = [float(prop) for prop in iris_properties]
        iris_label = row[-1]
        label_idx = self._get_idx(iris_label)
        read_line.extend(iris_properties)
        read_line.extend([iris_label, label_idx])
        return read_line

    def _process_row_split(self, row):
        """
        Import lines from train/valid/test split files.
        Imports line of "property_1, property_2, property_3, property_4, label_name, label_index"
        Returns list of [property_1, property_2, property_3, property_4, label_index]
        """
        read_line = []
        iris_properties = row[:4]
        iris_properties = [float(prop) for prop in iris_properties]
        iris_label = row[4]
        label_idx = int(row[-1])
        fetch_idx = self._get_idx(iris_label)
        read_line.extend(iris_properties)
        read_line.append(label_idx)
        return read_line


class TaskData(DataManager):
    def __init__(self, filepath, split):
        super().__init__(filepath, split)
        self.filepath = filepath
        self.split = split       # train/valid/test fractions, should sum to 1
        # self.num_classes = 0
        self.discard_header = True
        self.dataset_path = './data/task'

    def _process_row_raw(self, row):
        """
        Import lines from raw Iris data file.
        Imports line of "index, property_1, property_2, label_index"
        Returns list of [property_1, property_2, label_index]
        """
        read_line = []
        task_x = float(row[1])
        task_y = float(row[2])
        task_label = int(float(row[-1]))
        count = self._count_idx(task_label)  # count the number of classes
        read_line.extend([task_x, task_y, task_label])
        return read_line

    def _process_row_split(self, row):
        """
        Import lines from train/valid/test split files.
        Imports line of "property_1, property_2, label_index"
        Returns list of [property_1, property_2, label_index]
        """
        read_line = []
        task_x = float(row[0])
        task_y = float(row[1])
        task_label = int(float(row[-1]))
        count = self._count_idx(task_label)  # count the number of classes
        read_line.extend([task_x, task_y, task_label])
        return read_line
