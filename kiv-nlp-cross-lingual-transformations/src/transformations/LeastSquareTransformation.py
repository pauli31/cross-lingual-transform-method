from transformations.Transform import Transform

import numpy as np
import logging
import deprecation

import torch
import torch.nn as nn

logger = logging.getLogger("LeastSquareTransformation")
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')


class LeastSquareTransformation(Transform):
    """
        Implementation transformation using MSE, or least square error

        Pavel Priban, pribanp@kiv.zcu.cz,
        27.6.2020 -  ver 0.1
        9.7.2021  -  ver 0.2 Added pytorch implementation, deprecated numpy gradient descent

    """

    def __init__(self, method="analytical", verbose=True, use_bias=False,
                 n_iters=10000, lr=0.01, regularization=0.0, disable_shape_check=False,
                 batch_size=10000):
        """
        Init Least square transformation with Mean Squared Error

        # :param gradient_descent: if set to True gradient descent is used, otherwise analytical solution is applied
        :param method method used to obtain the transformation matrix, possible values : analytical, sgd, analytical_torch, sgd_torch
        :param verbose: if True logs will be printed
        :param use_bias:  whether to use bias, default False
        :param n_iters: number of iterations for gradient descent version
        :param lr: learning rate for gradient descent
        :param regularization: floating point number for reguralization, advised value is between 0 and 1
        :param disable_shape_check: disable shape check
        :param batch_size used only for torch versions of LST
        """

        super().__init__(verbose)
        self.method = method
        self.use_bias = use_bias
        self.n_iters = n_iters
        self.lr = lr
        self.regularization = regularization
        self.disable_shape_check = disable_shape_check
        self.batch_size = batch_size

        logger.info("LeastSquareTransformation initialized")

    @Transform.overrides(Transform)
    def transform(self, X_src, Y_trg):
        if self.disable_shape_check is False:
            self.check_same_shape(X_src, Y_trg)

        T = self.compute_T_mat(X_src, Y_trg)

        # add bias term
        if self.use_bias:
            X_src = self.add_bias(X_src)

        # transformed matrix
        X_t = X_src @ T
        return X_t

    @Transform.overrides(Transform)
    def compute_T_mat(self, X_src, Y_trg):
        if self.method == "sgd":
            raise Exception("This method is deprecated")
            T = self.compute_T_gradient(X_src, Y_trg)
        elif self.method == "analytical":
            T = self.comopute_T_analytical(X_src, Y_trg)
        elif self.method == "sgd_torch":
            T = self.compute_T_sgd_torch(X_src, Y_trg)
        elif self.method == "sgd_torch_2":
            T = self.compute_T_sgd_torch_v2(X_src, Y_trg)
        elif self.method == "analytical_torch":
            T = self.compute_T_torch_analytical(X_src, Y_trg)
        else:
            raise Exception("Unkonwn method:" + self.method)
        return T

    @Transform.overrides(Transform)
    def compute_error(self, X_src, Y_trg, T, add_bias=False):

        if add_bias is True:
            # add bias term
            if self.use_bias:
                X_src = self.add_bias(X_src)

        # return np.linalg.norm((X_src @ T) - Y_trg, ord=2) ** 2
        # return np.linalg.norm((X_src @ T) - Y_trg) ** 2

        return super(LeastSquareTransformation, self).compute_error(X_src, Y_trg, T)

    # Muselo by se to upravit pro matice funguje pouze pro vektory
    @deprecation.deprecated()
    def cost_function(self, X_src, Y_trg, T, m):
        # return (1/2*m) * np.transpose((X_src @ T - Y_trg)) @ (X_src @ T - Y_trg)
        J_reg = 0
        if self.regularization is not None and self.regularization != 0:
            lamb = self.regularization

            # because in case of using bias we don't want to incorporate
            # bias parameters into cost function
            index = 0
            if self.use_bias:
                index = 1
            J_reg = lamb * np.sum(np.transpose(T[index:]) @ T[index:])

        h = (X_src @ T) - Y_trg
        return (1./(2*m)) * (np.sum(np.transpose(h) @ h) + J_reg)

    def add_bias(self, X):
        # add ones to column
        X = np.c_[np.ones(X.shape[0]), X]
        return X

    def comopute_T_analytical(self, X_src, Y_trg):
        """
            it computes the transformation matrix by (X.T X)^-1 @ Y,
            and (X.T X)^-1 is the pseudoinverse matrix
        """

        mx, nx = X_src.shape

        if self.regularization == 0:
            X_pinv = np.linalg.pinv(X_src)
            T = X_pinv @ Y_trg
        else:
            I = np.eye((nx))
            T_trans = np.transpose(X_src)
            T = np.linalg.inv((T_trans @ X_src) + (self.regularization * I)) @ (T_trans @ Y_trg)

        return T

    # Muselo by se to upravit pro matice funguje pouze pro vektory
    @deprecation.deprecated()
    def compute_T_gradient(self, X_src, Y_trg):
        # add bias term
        if self.use_bias:
            X_src = self.add_bias(X_src)

        # get dimensions, n - embeddings dimension, m - number of examples (words)
        mx, nx = X_src.shape
        my, ny = Y_trg.shape

        if mx != my:
            raise Exception("mx is not equal to my, mx:" + str(mx) + " my:" + str(my))

        T = np.ones((nx, ny))
        iterations = self.n_iters
        lr = self.lr
        lamb = self.regularization

        # we want to distinguish when we use bias and when not
        index = 0
        if self.use_bias:
            index = 1

        for iteration in range(iterations):
            h = (X_src @ T) - Y_trg
            if self.use_bias:
                T[0] = T[0] - lr * ((1 / mx) * (np.transpose(X_src[:, 0]) @ h))

            T[index:] = T[index:] - lr * ((1 / mx) * ((np.transpose(X_src[:, index:]) @ h) + (lamb * T[index:])))

            if self.verbose is True:
                if iteration % 1000 == 0:
                    cost = self.cost_function(X_src, Y_trg, T, mx)
                    logger.info("iter :" + str(iteration) + " Cost:" + str("{:.8f}".format(cost)))

        return T

    # mainly based on
    # http://www.cse.chalmers.se/~richajo/dit866/backup_2019/lectures/l5/PyTorch%20linear%20regression%20demo.html
    # https://towardsdatascience.com/how-to-implement-linear-regression-with-pytorch-5737339296a6
    # and on our gradient descent implementation
    def compute_T_sgd_torch(self, X_src, Y_trg):

        # get dimensions, n - embeddings dimension, m - number of examples (words)
        mx, nx = X_src.shape
        my, ny = Y_trg.shape

        if mx != my:
            raise Exception("mx is not equal to my, mx:" + str(mx) + " my:" + str(my))

        Xt = torch.tensor(X_src, dtype=torch.float)
        Yt = torch.tensor(Y_trg, dtype=torch.float)

        # init the matrix to all zeroes
        T = torch.zeros((nx, ny), requires_grad=True, dtype=torch.float)

        history = []
        iterations = self.n_iters
        lr = self.lr
        batch_size = self.batch_size

        # we select an optimizer, in this case (minibatch) SGD.
        # it needs to be told what parameters to optimize, and what learning rate (lr) to use
        # we could use another one Adam for example
        optimizer = torch.optim.SGD([T], lr=lr)

        total_loss = 0

        for iteration in range(iterations):
            total_loss = 0

            for batch_count, batch_start in enumerate(range(0, nx, batch_size)):
                batch_end = batch_start + batch_size

                # pick out the batch again
                Xbatch = Xt[batch_start:batch_end, :]
                Ybatch = Yt[batch_start:batch_end, :]

                # compute the transformation, i.e. multiply X with T and substract target matrix
                # we get an hypotesis (error)
                h = torch.matmul(Xbatch, T) - Ybatch

                # compute loss for batch (same equation as cost_function function
                # This is wrong it works only for vectors it is not frobenius norm
                # loss_batch = (1./(2*n_examples)) * torch.matmul(torch.transpose(h, 0, 1), h)

                loss_batch = torch.sum(torch.mul(h, h))

                # sum the loss
                # loss_batch_value = loss_batch.item()
                loss_batch_value = torch.sum(loss_batch).item()

                total_loss += loss_batch_value

                # reset all gradients
                optimizer.zero_grad()

                # compute the gradients for the loss for this batch
                loss_batch.backward()

                # update parameters
                optimizer.step()

            if self.verbose is True:
                if iteration % 1000 == 0:
                    logger.info(70 * "-")
                    logger.info("iter :" + str(iteration) + "Sum Total Loss:" + str("{:.8f}".format(total_loss)))
                    logger.info(70 * "-")

            history.append(total_loss)

        logger.info("Total final loss:" + str("{:.8f}".format(total_loss)))
        T_np = T.detach().numpy()

        return  T_np

    def compute_T_sgd_torch_v2(self, X_src, Y_trg):
        # get dimensions, n - embeddings dimension, m - number of examples (words)
        mx, nx = X_src.shape
        my, ny = Y_trg.shape

        history = []

        if mx != my:
            raise Exception("mx is not equal to my, mx:" + str(mx) + " my:" + str(my))

        Xt = torch.tensor(X_src, dtype=torch.float)
        Yt = torch.tensor(Y_trg, dtype=torch.float)

        model = TorchNNLeastSquareTransformation(nx, ny, self.use_bias)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss(reduction='sum')

        total_loss = 0
        for iter in range(self.n_iters):
            total_loss = 0

            for batch_count, batch_start in enumerate(range(0, nx, self.batch_size)):
                batch_end = batch_start + self.batch_size

                # pick out the batch again
                Xbatch = Xt[batch_start:batch_end, :]
                Ybatch = Yt[batch_start:batch_end, :]

                pred = model(Xbatch, Ybatch)

                # Frobenius norm power 2, Frobeniusova norma umocnena na druhou - tzn. nemusim tam odmocnovat ani mocnit na druhou
                # na druhou se mocni akorat jednotlive slozky matice, a to delam tim mul.
                # loss_batch_2 = torch.mean(torch.mul(pred_err, pred_err))
                # loss_batch_3 = torch.sum(torch.mul(pred_err, pred_err))/n_examples
                loss_batch = loss_fn(pred, Ybatch)

                loss_batch_value = loss_batch.item()
                total_loss += loss_batch_value

                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()

            if self.verbose is True:
                if iter % 1000 == 0:
                    logger.info(70 * "-")
                    logger.info("iter :" + str(iter) + "Total Loss:" + str("{:.8f}".format(total_loss)))
                    logger.info(70 * "-")

            history.append(total_loss)

        logger.info("Total final loss:" + str("{:.8f}".format(total_loss)))
        T_np = model.T_matrix.weight.detach().numpy()
        T_np = T_np.T
        return T_np


    def compute_T_torch_analytical(self, X_src, Y_trg):
        """
            it computes the transformation matrix by (X.T X)^-1 @ Y,
            and (X.T X)^-1 is the pseudoinverse matrix
        """

        Xt = torch.tensor(X_src, dtype=torch.float)
        Yt = torch.tensor(Y_trg, dtype=torch.float)

        # https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html#torch.linalg.pinv
        # X_pinv = torch.pinverse(Xt)
        # T =  torch.matmul(X_pinv, Yt)

        # it is preferred to use lstsq
        T = torch.linalg.lstsq(Xt, Yt).solution

        return T.numpy()


class TorchNNLeastSquareTransformation(nn.Module):
    def __init__(self, nx, ny, use_bias):
        super(TorchNNLeastSquareTransformation, self).__init__()

        self.use_bias = use_bias
        self.T_matrix = nn.Linear(nx, ny, bias=use_bias)

    def forward(self, X_mat, Y_mat):
        h = self.T_matrix(X_mat)
        return h


if __name__ == '__main__':
    from tests.LeastSquareTransformationTest import mainLSTtest
    mainLSTtest()



