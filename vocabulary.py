"""
Class for building and managing vocabulary for NLP.
Based on some TensorFlow data_utils.
"""
import os.path
import re

# String to use for padding or unknown token
_PAD = "_PAD"
_UNK = "_UNK"

# Padding for 0 index, if used
START_VOCAB_dict = dict()
START_VOCAB_dict['with_padding'] = [_PAD, _UNK]
START_VOCAB_dict['no_padding'] = [_UNK]

# UNK ID depends if padding is used or not
UNK_ID_dict = dict()
UNK_ID_dict['with_padding'] = 1
UNK_ID_dict['no_padding'] = 0

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

class Vocabulary(object):
    def __init__(self, datadir, max_vocabulary_size):
        self.datadir = datadir
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)
        self.max_vocabulary_size = max_vocabulary_size
        self.corpus_file        = 'sentences_raw.txt'
        self.vocab_file         = 'vocab_sentences.txt'
        self.label_file         = 'vocab_labels.txt'

        # train/valid/test file names
        self.sentences_file     = 'sentences.txt'
        self.labels_file        = 'labels.txt'
        self.ids_sentences_file = 'ids_sentences.txt'
        self.ids_labels_file    = 'ids_labels.txt'

        self.tokeniser = self.basic_tokeniser

        self.vocab_list = []
        self.label_list = []

        self.vocab_to_id = {}
        self.id_to_vocab = []
        self.label_to_id = {}
        self.id_to_label = []

    def build_vocabulary(self, sentences, labels):
        self.build_sentence_vocabulary(sentences)
        self.build_label_vocabulary(labels)

    def build_sentence_vocabulary(self, sentences, normalise_digits=True):
        """Build vocabulary from a list of sentences"""
        # write sentences (corpus) to file
        sentences_raw_path = os.path.join(self.datadir, self.corpus_file)
        if not os.path.exists(sentences_raw_path):
            print('Creating sentence corpus in {}'.format(sentences_raw_path))
            self._write_file_from_list(sentences_raw_path, sentences)

        # Build vocabulary from list
        print("Building vocabulary")
        counter = 0
        vocab = {}  # Count occurrences of each word
        for sentence in sentences:
            counter += 1
            if counter % 5000 == 0:
                print("  processing line {}".format(counter))
            tokens = self.tokeniser(sentence)
            for w in tokens:
                word = re.sub(_DIGIT_RE, "0", w) if normalise_digits else w
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab_list = START_VOCAB_dict['with_padding'] + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > self.max_vocabulary_size:
            vocab_list = vocab_list[:self.max_vocabulary_size]
        self.vocab_list = vocab_list

        # Write vocabulary to file
        vocab_path = os.path.join(self.datadir, self.vocab_file)
        self._write_file_from_list(vocab_path, vocab_list)

    def build_label_vocabulary(self, labels):
        labels = sorted(set(labels))
        self.label_list = labels
        label_path = os.path.join(self.datadir, self.label_file)
        self._write_file_from_list(label_path, labels)

    def get_sentence_vocabulary(self):
        vocab, rev_vocab = self.initialise_vocabulary(self.vocab_list)
        self.vocab_to_id = vocab
        self.id_to_vocab = rev_vocab
        return vocab, rev_vocab

    def get_label_vocabulary(self):
        vocab, rev_vocab = self.initialise_vocabulary(self.label_list)
        self.label_to_id = vocab
        self.id_to_label = rev_vocab
        return vocab, rev_vocab

    def initialise_vocabulary(self, vocabulary_list):
        """
        Initialise vocabulary from vocabulary list
        Build Vocab > ID dictionary, and ID > Vocab list.
        """
        rev_vocab = [line.strip() for line in vocabulary_list]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab

    def basic_tokeniser(self, sentence):
        """Very basic tokenizer: split the sentence into a list of tokens."""
        words = []
        for space_separated_fragment in sentence.strip().split():
            words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
        return [w for w in words if w]

    def sentence_to_token_ids(self, sentence, UNK_ID, normalise_digits=True):
        words = self.tokeniser(sentence)
        if not normalise_digits:
            return [self.vocab_to_id.get(w, UNK_ID) for w in words]
        # Normalize digits by 0 before looking words up in the vocabulary.
        return [self.vocab_to_id.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]

    def data_to_token_ids(self, data, split_name, use_padding=True, normalise_digits=True):
        """
        Converts a list of data into its token IDs, writes
        them to a file, and returns the ID form of the data
        """
        tokenised_data = []
        for i, line in enumerate(data):
            if i % 5000 == 0 and i != 0:
                print('  tokenising line {}'.format(i))
            if use_padding:
                UNK_ID = UNK_ID_dict['with_padding']
            else:
                UNK_ID = UNK_ID_dict['no_padding']
            token_ids = self.sentence_to_token_ids(line, UNK_ID, normalise_digits)
            tokenised_data.append(token_ids)

        # write file
        write_dir = os.path.join(self.datadir, split_name)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        sents_file     = os.path.join(write_dir, self.sentences_file)
        ids_sents_file = os.path.join(write_dir, self.ids_sentences_file)

        self._write_file_from_list(sents_file, data)
        self._write_file_from_list(ids_sents_file, tokenised_data)
        return tokenised_data

    def labels_to_token_ids(self, labels, split_name):
        tokenised_labels = []
        for label in labels:
            tokenised_labels.append(self.label_to_id[label])

        # write file
        write_dir = os.path.join(self.datadir, split_name)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        labels_file     = os.path.join(write_dir, self.labels_file)
        ids_labels_file = os.path.join(write_dir, self.ids_labels_file)

        self._write_file_from_list(labels_file, labels)
        self._write_file_from_list(ids_labels_file, tokenised_labels)
        return tokenised_labels

    def translate_examples(self, examples):
        for sentence_tokens, label_token in examples:
            sentence_list = []
            for token in sentence_tokens:
                sentence_list.append(self.id_to_vocab[token])
            sentence = " ".join(sentence_list)
            label = self.id_to_label[label_token]
            print("{}\n{}\n\n".format(sentence, label))


    def _write_file_from_list(self, filename, write_list):
        print('Writing {} ...'.format(filename))
        with open(filename,'w') as file:
            for write_line in write_list:
                if not isinstance(write_line, (str, int)):
                    write_line = " ".join([str(item) for item in write_line])
                file.write(str(write_line) + "\n")
