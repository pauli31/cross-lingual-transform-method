import logging
import numpy as np

# https://realpython.com/python-super/

# http://www.cse.chalmers.se/~richajo/dit866/backup_2019/lectures/l5/PyTorch%20linear%20regression%20demo.html - pytorch implementace
# https://towardsdatascience.com/how-to-implement-linear-regression-with-pytorch-5737339296a6
# https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
# https://discuss.pytorch.org/t/how-does-one-implement-weight-regularization-l1-or-l2-manually-without-optimum/7951
# https://discuss.pytorch.org/t/simple-l2-regularization/139 - reguralizace

class Transform(object):
    """
          Generally,
          T denotes the transformation matrix
          X denotes the source matrix
          Y denotes the target matrix

          usually the the transformation is then done by
          X_t = X @ T

          which transforms X into space of Y, and @ is matrix multiplication
      """

    def __init__(self, verbose):
        self.verbose = verbose

        if verbose is True:
            logging.root.setLevel(level=logging.INFO)
        else:
            logging.root.setLevel(level=logging.ERROR)

    def compute_T_mat(self, X_src, Y_trg):
        """
        Computes the transformation transformation matrix.
        The X_src and Y_trg must have the same dimension and be row-wise aligned.
        Because all row vectors are used as link in the transformation

        :param X_src: source space (ndarray) with shape (number of vectors, embeddings dimension)
        :param Y_trg: target space (ndarray) with shape (number of vectors, embeddings dimension)

        :return: transformation matrix (ndarray) with shape (embeddings dimension, embeddings dimension)
        """
        raise NotImplementedError("Not implemented")

    def transform(self, X_src, Y_trg):
        """
        Transform X_src matrix into Y_trg, the shape of matrices must be same,
        and row vectors must be aligned, because all vectors from both matrices
        are used for transformation.

        This default implementation will call method compute_T_mat() and then use
        this matrix to transform the X_src space as follows:
        X_t = X_src @ T

        :param X_src: source embedding matrix (ndarray) with shape (number of vectors, embeddings dimension)
        :param Y_trg: target embedding matrix (ndarray) with shape (number of vectors, embeddings dimension)

        :return: the transformed space (ndarray) of X_src into trg_emb space, with shape (number of vectors, embeddings
                dimension)
        """

        self.check_same_shape(X_src, Y_trg)

        T = self.compute_T_mat(X_src, Y_trg)
        X_t = X_src @ T

        return X_t

    def transform_with_T(self, X_src, T_matrix):
        """
        Transforms given source space with a transformation matrix.
        Usually it multiplies the X_src with the T_matrix

        :param X_src: source space (ndarra) with shape (number of vectors, embeddings dimension)
        :param T_matrix: transformation matrix with shape (embeddings dimension, embeddings dimension)

        :return: the transformed space (ndarray) with shape (number of vectors, embeddings dimension)
        """
        if X_src.shape[1] != T_matrix.shape[0]:
            raise Exception("Shapes do not match: " + str(X_src.shape[1]) + " T_matrix.shape[0]: "+ str(T_matrix.shape[0]))

        X_t = X_src @ T_matrix
        return X_t

    def compute_error(self, X_src, Y_trg, T):
        """
        Computes how far is the X after transformation into Y using the T matrix
        The default implementation uses the MSE as shown in README

        :param X_src: source space (ndarray) with shape (number of vectors, embeddings dimension)
        :param Y_trg: arget space (ndarray) with shape (number of vectors, embeddings dimension)
        :param T: transformation matrix with shape (embeddings dimension, embeddings dimension)
        :return: a value of error
        """
        return np.linalg.norm((X_src @ T) - Y_trg) ** 2

    @staticmethod
    def check_same_shape(X_src, Y_trg):
        if X_src.shape != Y_trg.shape:
            raise Exception("Shapes of the input embeddings does not match, X_src shape:" + str(X_src.shape) +
                            " Y_trg shape:" + Y_trg.shape)

    def overrides(interface_class):
        def overrider(method):
            assert (method.__name__ in dir(interface_class))
            return method

        return overrider