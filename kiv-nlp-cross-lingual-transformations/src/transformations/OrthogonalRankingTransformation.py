from transformations.Transform import Transform

import logging
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')


class OrthogonalRankingTransformation(Transform):
    """
        Implementation of Orthogonal Ranking Transformation

        Adam Mistera, amistera@kiv.zcu.cz

        23.9.2021 -  ver 0.1
    """

    def __init__(self,
                 verbose=True,
                 use_bias=False,
                 negative_examples=2,
                 delta=2,
                 lr=0.001,
                 regularization=0.0,
                 iterations=25,
                 batch_size=1000,
                 disable_shape_check=False):
        """
        Init Orthogonal Ranking Transformation

        :param verbose: flag if logs will be printed, default True
        :param use_bias: flag if bias will be used, default False
        :param negative_examples: number of negative examples used for transformation for each side
        :param delta: distance that correct transformation of vector should have from negative example
        :param lr: floating point number representing learning rate for gradient descent
        :param regularization: floating point number representing regularization parameter
        :param iterations: number of iterations used for learning
        :param batch_size: batch size used in torch
        :param disable_shape_check: flag if shape will be checked, default False
        """

        super().__init__(verbose)
        self.use_bias = use_bias
        self.negative_examples = negative_examples
        self.delta = delta
        self.lr = lr
        self.regularizaion = regularization
        self.iterations = iterations
        self.batch_size = batch_size
        self.disable_shape_check = disable_shape_check

        logger.info("OrthogonalRankingTransformation initialized")

    @Transform.overrides(Transform(True))
    def transform(self, X_src, Y_trg):
        T = self.compute_T_mat(X_src, Y_trg)
        X_t = X_src @ T
        return X_t

    @Transform.overrides(Transform(True))
    def compute_T_mat(self, X_src, Y_trg):
        T = self.compute_T(X_src, Y_trg)
        return T

    def compute_T(self, X_src, Y_trg):
        mx, nx = X_src.shape
        my, ny = Y_trg.shape

        X = torch.tensor(X_src, dtype=torch.float32)
        Y = torch.tensor(Y_trg, dtype=torch.float32)
        T = torch.zeros((nx, ny), requires_grad=True, dtype=torch.float32)


        iterations = self.iterations
        batch_size = self.batch_size
        negative_examples = self.negative_examples
        delta = torch.tensor(self.delta, dtype=torch.float32)

        optimizer = torch.optim.Adam([T], lr=self.lr)
        total_loss, cosine, cosine_backwards = 0.0, [], []

        for iteration in range(iterations):
            for batch_start in range(0, X.size()[0], batch_size):
                X_batch = X[batch_start:batch_start + batch_size]
                Y_batch = Y[batch_start:batch_start + batch_size]

                # Predict values for Y and backwards for X
                X_pred = torch.matmul(Y_batch, T.T)
                Y_pred = torch.matmul(X_batch, T)

                # Get negative examples for X
                X_distances = torch.cdist(X_pred, X_batch) - torch.cdist(X_batch, X_batch)
                X_distances.fill_diagonal_(float('inf'))  # i != j
                X_negatives = X_batch[torch.squeeze(X_distances.topk(negative_examples, largest=False).indices)]

                # Get negative examples for Y
                Y_distances = torch.cdist(Y_pred, Y_batch) - torch.cdist(Y_batch, Y_batch)
                Y_distances.fill_diagonal_(float('inf'))  # i != j
                Y_negatives = Y_batch[torch.squeeze(Y_distances.topk(negative_examples, largest=False).indices)]

                # Compute loss value using hinge loss function
                # Sum over all negative examples from both sides
                negatives_x = delta + torch.linalg.norm(X_pred - X_batch, dim=1)[:, None] - torch.linalg.norm(
                    X_pred[:, None] - X_negatives, dim=2)
                negatives_y = delta + torch.linalg.norm(Y_pred - Y_batch, dim=1)[:, None] - torch.linalg.norm(
                    Y_pred[:, None] - Y_negatives, dim=2)
                batch_loss = torch.sum(torch.maximum(torch.tensor(0.0, dtype=torch.float32), negatives_x)) + \
                             torch.sum(torch.maximum(torch.tensor(0.0, dtype=torch.float32), negatives_y))

                # Update total loss
                total_loss += batch_loss

                # Compute cosine similarity of the batch
                cosine.append(torch.mean(F.cosine_similarity(Y_pred, Y_batch)).detach().numpy())

                # Compute cosine similarity of the batch backwards
                cosine_backwards.append(torch.mean(F.cosine_similarity(X_pred, X_batch)).detach().numpy())

                # Reset all gradients
                optimizer.zero_grad()

                # Compute the gradients for the loss for this batch
                batch_loss.backward()

                # Update parameters
                optimizer.step()

            if self.verbose is True:
                logger.info(110 * "-")
                logger.info(f"Iteration: {iteration}\tSum Total Loss: {total_loss:.5f}\tCosine similarity: "
                            f"{np.mean(cosine):.4f}\tCosine similarity backwards: {np.mean(cosine_backwards):.4f}")
                logger.info(110 * "-")

            total_loss, cosine, cosine_backwards = 0.0, [], []

        return T.detach().numpy()

    def __str__(self):
        return "OrthogonalRankingTransformation"


if __name__ == '__main__':
    from tests.OrthogonalRankingTransformationTest import mainORTtest
    mainORTtest()
