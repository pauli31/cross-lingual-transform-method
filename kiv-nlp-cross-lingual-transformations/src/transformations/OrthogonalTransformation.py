from transformations.Transform import Transform

import numpy as np
import logging

import torch

logger = logging.getLogger("OrthogonalTransformation")
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')


class OrthogonalTransformation(Transform):
    def __init__(self, method='numpy', verbose=True):
        super().__init__(verbose)
        """
        :param method: method used to analytical computation, possible valuse: numpy, torch
        """

        self.method = method
        logger.info("OrthogonalTransformation initialized")

    @Transform.overrides(Transform)
    def compute_T_mat(self, X_src, Y_trg):
        if self.method == 'numpy':
            T = self.compute_T_mat_numpy(X_src, Y_trg)
        elif self.method == 'torch':
            T = self.compute_T_mat_torch(X_src, Y_trg)
        else:
            raise Exception("Unknown method:" + str(self.method))

        return T

    def compute_T_mat_torch(self, X_src, Y_trg):
        self.check_same_shape(X_src, Y_trg)

        Xt = torch.tensor(X_src, dtype=torch.float)
        Yt = torch.tensor(Y_trg, dtype=torch.float)

        u, s, v = torch.linalg.svd(torch.matmul(Yt.t(), Xt), full_matrices=False)

        T = torch.matmul(v.t(), u.t())
        T = T.numpy()
        return T

    def compute_T_mat_numpy(self, X_src, Y_trg):
        self.check_same_shape(X_src, Y_trg)

        # compute svd - USV
        u, s, v = np.linalg.svd((np.transpose(Y_trg) @ X_src), full_matrices=False)

        T = np.transpose(v) @ np.transpose(u)
        return T

if __name__ == '__main__':
    from tests.OrthogonalTransformationTest import mainOTTest
    mainOTTest()
