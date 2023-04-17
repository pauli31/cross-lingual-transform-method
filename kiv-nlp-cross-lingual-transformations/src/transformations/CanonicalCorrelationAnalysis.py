


import numpy as np
import logging
import torch

from transformations.Transform import Transform

logger = logging.getLogger("CCATransformation")
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')


class CanonicalCorrelationAnalysis(Transform):
    def __init__(self, method='numpy', verbose=True):
        """
        :param method: method used to analytical computation, possible valuse: numpy, torch
        """
        super().__init__(verbose)

        self.method = method
        logger.info("CCA Transformation initialized")

    @Transform.overrides(Transform)
    def compute_T_mat(self, X_src, Y_trg):
        self.check_same_shape(X_src, Y_trg)
        if self.method == 'numpy':
            A, B = self.compute_CCA_transform_matrices_numpy(X_src, Y_trg)
        elif self.method == 'torch':
            A, B = self.compute_CCA_transform_matrices_torch(X_src, Y_trg)
        else:
            raise Exception("Unknown method:" + str(self.method))
        # transformation matrix from x to y
        T = A @ np.linalg.pinv(B)
        return T

    def compute_CCA_transform_matrices_torch(self, X_src, Y_trg):
        self.check_same_shape(X_src, Y_trg)

        Xt = torch.tensor(X_src, dtype=torch.float)
        Yt = torch.tensor(Y_trg, dtype=torch.float)

        full = False

        # compute svd for X, x = ux @ np.diag(sx) @ vx, vhx is transposed vx
        # full_matrices=False - we want only the reduced
        ux, sx, vhx = torch.linalg.svd(Xt, full_matrices=full)

        # compute svd for Y, x = uy @ np.diag(s) @ vy
        uy, sy, vhy = torch.linalg.svd(Yt, full_matrices=full)

        # get matrix O
        O = torch.matmul(ux.t(), uy)

        # turn 1-D into 2-D tensor, into a diagonal matrix from array
        sx = torch.diag(sx)
        sy = torch.diag(sy)

        sxinv = torch.linalg.pinv(sx)
        syinv = torch.linalg.pinv(sy)

        # get svd of O
        uo, so, vho = torch.linalg.svd(O, full_matrices=full)

        # for x
        A = vhx.t() @  sxinv @ uo

        # for y
        B = vhy.t() @ syinv @ vho.t()

        A_np = A.numpy()
        B_np = B.numpy()

        return A_np, B_np

    def compute_CCA_transform_matrices_numpy(self, X_src, Y_trg):
        """
        Computes transformation matrices into third space O for both given spaces.
        Both given matrices must have the same dimension

        :param X_src: source space (ndarray) with shape (num of vectors, embeddings size)
        :param Y_trg: target space (ndarray) with shape (num of vectors, embeddings size)

        :return: matrices A and B that transforms given spaces into third shared space.
                 A - a transformation matrix (ndarray) with shape (embeddings size, embeddings size) for X_src space
                 B - a transformation matrix (ndarray) with shape (embeddings size, embeddings size) for Y_trg space
        """
        self.check_same_shape(X_src, Y_trg)

        # compute svd for X, x = ux @ np.diag(sx) @ vx, vhx is transposed vx
        # full_matrices=False - we want only the reduced
        full = False
        ux, sx, vhx = np.linalg.svd(X_src, full_matrices=full)

        # compute svd for Y, x = uy @ np.diag(s) @ vy
        uy, sy, vhy = np.linalg.svd(Y_trg, full_matrices=full)

        # turn the diagonal vectors into m by m matrices
        sx = np.diag(sx)
        sy = np.diag(sy)

        # get matrix O
        O = np.transpose(ux) @ uy
        # get svd of O
        uo, so, vho = np.linalg.svd(O, full_matrices=full)

        sxinv = np.linalg.pinv(sx)
        syinv = np.linalg.pinv(sy)

        # for x
        A = np.transpose(vhx) @ sxinv @ uo

        # for y
        B = np.transpose(vhy) @ syinv @ np.transpose(vho)

        return A, B


if __name__ == '__main__':
    from tests.CanonicalCorrelationAnalysisTest import mainCCATest
    mainCCATest()




