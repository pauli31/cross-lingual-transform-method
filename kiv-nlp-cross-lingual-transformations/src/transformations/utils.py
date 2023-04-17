import numpy as np
from numpy.linalg import norm
import logging
import os

from tests.config import ANALOGIES_DIR

logger = logging.getLogger("utils")
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')


def normalize(X, mean_centering=True, unit_vectors=True, unit_vec_features=False):
    """
    Normalize given ndarray

    :param X: ndarray representing semantic space,
                axis 0 (rows)       - vectors for words
                axis 1 (columns)    - elements of word vectors
    :param mean_centering: if true values are centered around zero
    :param unit_vectors: is true vectors are converted to unit vectors

    :return: normalized ndarray
    """
    if mean_centering is True:
        # mean vector and normalization
        mean = X.sum(0) / X.shape[0]
        X = X - mean

    if unit_vectors is True:
        # compute norm
        # norms = np.sqrt((X ** 2).sum(-1))[..., np.newaxis]
        if unit_vec_features is False:
            # normalization over vectors of words
            norms = np.linalg.norm(X, axis=1).reshape((X.shape[0], 1))
        else:
            # normalization over features (columns)
            norms = np.linalg.norm(X, axis=0).reshape((1, X.shape[1]))
        X = X / norms
    return X

def normalize_custom(X, mean_centering=True, unit_vectors=True, unit_vec_features=False):
    """
    Normalize given ndarray

    :param X: ndarray representing semantic space,
                axis 0 (rows)       - vectors for words
                axis 1 (columns)    - elements of word vectors
    :param mean_centering: if true values are centered around zero
    :param unit_vectors: is true vectors are converted to unit vectors

    :return: normalized ndarray
    """
    mean = None
    norms = None

    if mean_centering is True:
        # mean vector and normalization
        mean = X.sum(0) / X.shape[0]
        X = X - mean

    if unit_vectors is True:
        # compute norm
        # norms = np.sqrt((X ** 2).sum(-1))[..., np.newaxis]
        if unit_vec_features is False:
            # normalization over vectors of words
            norms = np.linalg.norm(X, axis=1).reshape((X.shape[0], 1))
        else:
            # normalization over features (columns)
            norms = np.linalg.norm(X, axis=0).reshape((1, X.shape[1]))
        X = X / norms

    return X, mean, norms


class NormalizedFastText(object):

    def __init__(self, fasttext, embeddings_size, mean_centering=True, unit_vectors=True, unit_vec_features=False,
                 disable_normalization=False):
        self.fasttext = fasttext
        self.disable_normalization = disable_normalization
        self.embeddings_size = embeddings_size
        self.mean_centering = mean_centering
        self.unit_vectors = unit_vectors
        self.unit_vec_features = unit_vec_features
        self.words_set = set(self.fasttext.get_words())
        # self.normalized_X, self.normalized_mean, self.normalized_norms = self.compute_normalization()
        _, self.normalized_mean, self.normalized_norms = self.compute_normalization()



    def compute_normalization(self):
        words = self.fasttext.get_words()
        words_size = len(words)

        X = np.zeros(shape=(words_size, self.embeddings_size))

        for i, word in enumerate(words) :
            index = self.fasttext.get_word_id(word)
            if index != i:
                raise Exception("Failed check index:" + str(index) + " i:" + str(i) + " word:" + str(word))
            X[i] = self.fasttext[word]

        X_norm, mean, norms = normalize_custom(X, mean_centering=self.mean_centering,
                                        unit_vectors=self.unit_vectors,
                                        unit_vec_features=self.unit_vec_features)
        return X_norm, mean, norms

    def get_dimension(self):
        return self.fasttext.get_dimension()

    def get_words(self, include_freq=False, on_unicode_error='strict'):
        return self.fasttext.get_words(include_freq=include_freq, on_unicode_error=on_unicode_error)

    def get_word_vector(self, word):
        # if word in self.words_set:
        #     word_id = self.fasttext.get_word_id(word)
        #     vector = self.normalized_X[word_id]
        # else:
        #     vector = self.fasttext.get_word_vector(word)
        #
        #     if self.mean_centering:
        #         vector = vector - self.normalized_mean
        #
        #     if self.unit_vectors is True:
        #         if self.unit_vec_features is False:
        #             norm = np.linalg.norm(vector)
        #             vector = vector / norm
        #         else:
        #             vector = vector / self.normalized_norms

        vector = self.fasttext.get_word_vector(word)

        if self.disable_normalization is False:
            if self.mean_centering:
                vector = vector - self.normalized_mean

            if self.unit_vectors is True:
                if self.unit_vec_features is False:
                    norm = np.linalg.norm(vector)
                    vector = vector / norm
                else:
                    vector = vector / self.normalized_norms

        return vector

    @property
    def words(self):
        if self._words is None:
            self._words = self.get_words()
        return self._words

    def __getitem__(self, word):
        return self.get_word_vector(word)

    def __contains__(self, word):
        return word in self.words


def build_transform_vectors_fasttext(X, Y, mode, x_word2id, y_word2id, word_pairs, excluded=None, max=None):
    tuple_list = None
    # make pairs as an intersection of vocabularies
    if mode == 'intersection':
        # in this case we ignore parameter word_pairs

        x_word_set = set(x_word2id.values())
        y_word_set = set(y_word2id.values())

        tuple_list = []
        inter = x_word_set.intersection(y_word_set)
        for word in inter:
            tuple_list.append((word, word))

    elif mode == 'list':
        if type(word_pairs) is list:
            tuple_list = word_pairs
        else:
            # read pairs from file
            with open(word_pairs, 'r', encoding='utf-8') as f:
                tuple_list = []
                for line in f:
                    line = line.strip().split()
                    if len(line) != 2:
                        logger.info("Bad format of line:" + str(line))
                        continue

                    tuple_list.append((line[0], line[1]))
    else:
        raise Exception("Unknown mode")

    lx = []
    ly = []
    final_list = []
    count = 0
    count_missing = 0
    # now filter the words
    for pair in tuple_list:
        x_word = pair[0]
        y_word = pair[1]

        if not x_word.strip() or not y_word.strip():
            continue

        if excluded is not None:
            if (x_word in excluded) or (y_word in excluded):
                continue

        if not x_word in x_word2id:
            count_missing += 1
            continue
        if not y_word in y_word2id:
            count_missing += 1
            continue

        final_list.append(pair)
        lx.append(X[x_word])
        ly.append(Y[y_word])

        count += 1
        if max is not None:
            if (mode == 'list') and (count >= max):
                break

    logger.info("We found:" + str(count))
    logger.info("We did not found:" + str(count))

    X_new = np.ndarray((len(lx), len(lx[0])), dtype=np.float32)
    Y_new = np.ndarray((len(ly), len(ly[0])), dtype=np.float32)
    for i, (x, y) in enumerate(zip(lx, ly)):
        X_new[i] = x
        Y_new[i] = y

    return X_new, Y_new, final_list


def build_transform_vectors(X, Y, mode, x_word2id, y_word2id, word_pairs, excluded=None, max=None):
    """
    Builds a transformation matrices

    :param X: first matrix
    :param Y: second matrix
    :param mode: one of 'intersection', 'list'
            if 'intersection'
             -  then parameter word_pairs is ignored,
                the resulting matrices are created as from vectors of words,
                that are common for both spaces

            if 'list'
            -   then vectors of given words are used for building
                the resulting matrices, the words are given by word_pairs parameter

    :param word_pairs: either list of tuples of words, or path to file
            with word pairs, where one pair per line separated by tabulator or space

    :param x_word2id: dictionary, key - word, value index into X
    :param y_word2id: dictionary, key - word, value index into Y
    :param excluded: list of words that will be excluded from building the resulting matrices
    :param max: max of first n words that will be used, only valid for mode 'list'
    :return: X_new, Y_new, final_list
            X_new - new X matrix
            Y_new - new Y matrix
            final_list - list of pairs of words that were used in the new matrices

    """

    tuple_list = None
    # make pairs as an intersection of vocabularies
    if mode == 'intersection':
        # in this case we ignore parameter word_pairs

        x_word_set = set(x_word2id.values())
        y_word_set = set(y_word2id.values())

        tuple_list = []
        inter = x_word_set.intersection(y_word_set)
        for word in inter:
            tuple_list.append((word, word))

    elif mode == 'list':
        if type(word_pairs) is list:
            tuple_list = word_pairs
        else:
            # read pairs from file
            with open(word_pairs, 'r', encoding='utf-8') as f:
                tuple_list = []
                for line in f:
                    line = line.strip().split()
                    if len(line) != 2:
                        logger.info("Bad format of line:" + str(line))
                        continue

                    tuple_list.append((line[0], line[1]))
    else:
        raise Exception("Unknown mode")

    lx = []
    ly = []
    final_list = []
    count = 0
    count_missing = 0
    # now filter the words
    for pair in tuple_list:
        x_word = pair[0]
        y_word = pair[1]

        if not x_word.strip() or not y_word.strip():
            continue

        if excluded is not None:
            if (x_word in excluded) or (y_word in excluded):
                continue

        if not x_word in x_word2id:
            count_missing +=1
            continue
        if not y_word in y_word2id:
            count_missing += 1
            continue

        final_list.append(pair)
        lx.append(X[x_word2id[x_word]])
        ly.append(Y[y_word2id[y_word]])

        count += 1
        if max is not None:
            if (mode == 'list') and (count >= max):
                break

    logger.info("We found:" + str(count))
    logger.info("We did not found:" + str(count))

    X_new = np.ndarray((len(lx), len(lx[0])), dtype=np.float32)
    Y_new = np.ndarray((len(ly), len(ly[0])), dtype=np.float32)
    for i, (x, y) in enumerate(zip(lx, ly)):
        X_new[i] = x
        Y_new[i] = y

    return X_new, Y_new, final_list


def compute_vec_len(a):
    """
    Computes norm of a given vector

    :param a: vector
    :return: norm of the vector
    """
    return np.linalg.norm(a)


def compute_cosine_similarity(a, b):
    return 1 - compute_cosine(a, b)


def compute_cosine(a, b):
    return (a @ b.T) / (norm(a) * norm(b))


def load_additional_words(lang,delimiter='\t'):
    base_folder = ANALOGIES_DIR
    categories_list = os.listdir(base_folder)
    categories = len(categories_list)
    words = []
    for category in categories_list:
        filename = os.path.join(base_folder, category, f"{lang}.txt")

        with open(filename, encoding='utf-8') as file:
            for line in file:
                if len(line.rstrip()) == 0 or delimiter not in line:
                    continue

                source, target = line.rstrip().split(delimiter)
                source = source.lower()
                target = target.lower()
                words.append(source)
                words.append(target)

    return words

def save_w2v_format_fasttext(normalized_fasttext, path, additional_words=None):
    # get all words from model
    fasttext_f = normalized_fasttext.fasttext
    words = fasttext_f.get_words()
    if additional_words is not None:
        words.extend(additional_words)
        words = set(words)
        words = list(words)

    with open(path, 'w') as file_out:

        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(fasttext_f.get_dimension()) + "\n")

        # line by line, you append vectors to VEC file
        for w in words:
            v = normalized_fasttext.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr + '\n')
            except:
                pass