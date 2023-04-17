from pytest import approx
import numpy as np
from sklearn.linear_model import LinearRegression
import os

from tests.config import EMB_TEST, LINKS_DIR
from LeastSquareTransformation import LeastSquareTransformation
from tests.CrossLingualTest import default_czech_english_test
from tests.testing_utils import load_word_vectors
from utils import build_transform_vectors, normalize


def stupid_example_test():
    print("Running stupid example")

    # stupid linear test
    X = np.array([[1, 1],
                  [1, 2],
                  [1, 3]])
    y = np.array([[1],
                  [2],
                  [3]])

    print("Running LST Torch analytical solution")
    mse = LeastSquareTransformation(method='analytical_torch', disable_shape_check=True)
    T = mse.compute_T_mat(X, y)
    assert T[0][0] == approx(0, abs=1e-3)
    assert T[1][0] == approx(1, abs=1e-3)

    test_err = mse.compute_error(X, y, T)
    print("Torch analytical test error:" + str("{:.4f}".format(test_err)))
    print("Torch analytical solution:" + str(T))

    # Torch SGD ver2
    mse = LeastSquareTransformation(method='sgd_torch_2', disable_shape_check=True)
    T = mse.compute_T_mat(X, y)
    assert T[0][0] == approx(0, abs=1e-3)
    assert T[1][0] == approx(1, abs=1e-3)

    test_err = mse.compute_error(X, y, T)
    print("Torch SGD ver2 test error:" + str("{:.4f}".format(test_err)))
    print("Torch SGD ver2 solution:" + str(T))

    # Torch sgd
    mse = LeastSquareTransformation(method='sgd_torch',disable_shape_check=True)
    T = mse.compute_T_mat(X, y)
    assert T[0][0] == approx(0, abs=1e-3)
    assert T[1][0] == approx(1, abs=1e-3)

    test_err = mse.compute_error(X, y, T)
    print("Torch SGD Stupid example test error:" + str("{:.4f}".format(test_err)))
    print("Torch SGD Stupid example solution:" + str(T))

    # Analytical
    mse = LeastSquareTransformation(verbose=False, disable_shape_check=True)
    T = mse.compute_T_mat(X, y)
    assert T[0][0] == approx(0, abs=1e-3)
    assert T[1][0] == approx(1, abs=1e-3)

    test_err = mse.compute_error(X, y, T)
    assert test_err == approx(0, abs=1e-3)
    print("Analytical Stupid example test error:" + str("{:.4f}".format(test_err)))
    print("Analytical Stupid example solution:" + str(T))

    # # Gradient without bias
    # mse = LeastSquareTransformation(method='sgd', use_bias=False, verbose=False,
    #                                 disable_shape_check=True)
    # T = mse.compute_T_mat(X, y)
    # assert T[0][0] == approx(0, abs=1e-3)
    # assert T[1][0] == approx(1, abs=1e-3)
    #
    # test_err = mse.compute_error(X, y, T)
    # assert test_err == approx(0, abs=1e-3)
    # print("Gradient without bias Stupid example test error:" + str("{:.4f}".format(test_err)))
    # print("Gradient without bias Stupid example solution:" + str(T))
    #
    # # Gradient without bias, regularization = 0.0
    # mse = LeastSquareTransformation(method='sgd', use_bias=False, verbose=False, regularization=0.0,
    #                                 disable_shape_check=True)
    # T = mse.compute_T_mat(X, y)
    # assert T[0][0] == approx(0, abs=1e-3)
    # assert T[1][0] == approx(1, abs=1e-3)
    #
    # test_err = mse.compute_error(X, y, T)
    # assert test_err == approx(0, abs=1e-3)
    # print("Gradient without bias, regularization = 0.0 Stupid example test error:" + str("{:.4f}".format(test_err)))
    # print("Gradient without bias, regularization = 0.0 Stupid example solution:" + str(T))
    #
    # # Gradient without bias, regularization = 0.0001
    # mse = LeastSquareTransformation(method='sgd', use_bias=False, verbose=False, regularization=0.0001,
    #                                 disable_shape_check=True)
    # T = mse.compute_T_mat(X, y)
    # assert T[0][0] == approx(0, abs=1e-3)
    # assert T[1][0] == approx(1, abs=1e-3)
    #
    # test_err = mse.compute_error(X, y, T)
    # assert test_err == approx(0, abs=1e-3)
    # print("Gradient without bias regularization = 0.0001 Stupid example test error:" + str("{:.4f}".format(test_err)))
    # print("Gradient without bias regularization = 0.0001 Stupid example solution:" + str(T))
    #
    # # Gradient with bias
    # mse = LeastSquareTransformation(method='sgd', use_bias=True, verbose=False, disable_shape_check=True)
    # T = mse.compute_T_mat(X, y)
    # assert T[0][0] == approx(0, abs=1e-3)
    # assert T[1][0] == approx(0, abs=1e-3)
    # assert T[2][0] == approx(1, abs=1e-3)
    #
    # test_err = mse.compute_error(X, y, T)
    # assert test_err == approx(0, abs=1e-3)
    # print("Gradient with bias Stupid example test error:" + str("{:.4f}".format(test_err)))
    # print("Gradient with bias Stupid example solution:" + str(T))
    #
    # # Gradient with bias, regularization = 0.0
    # mse = LeastSquareTransformation(method='sgd', use_bias=True, verbose=False, regularization=0.0,
    #                                 disable_shape_check=True)
    # T = mse.compute_T_mat(X, y)
    # assert T[0][0] == approx(0, abs=1e-3)
    # assert T[1][0] == approx(0, abs=1e-3)
    # assert T[2][0] == approx(1, abs=1e-3)
    #
    # test_err = mse.compute_error(X, y, T)
    # assert test_err == approx(0, abs=1e-3)
    # print("Gradient with bias Stupid example test error:" + str("{:.4f}".format(test_err)))
    # print("Gradient with bias Stupid example solution:" + str(T))
    #
    # # Gradient with bias, regularization = 0.0001
    # mse = LeastSquareTransformation(method='sgd', use_bias=True, verbose=False, regularization=0.0001,
    #                                 disable_shape_check=True)
    # T = mse.compute_T_mat(X, y)
    # assert T[0][0] == approx(0, abs=1e-3)
    # assert T[1][0] == approx(0, abs=1e-3)
    # assert T[2][0] == approx(1, abs=1e-3)
    #
    # test_err = mse.compute_error(X, y, T)
    # assert test_err == approx(0, abs=1e-3)
    # print("Gradient with bias Stupid example test error:" + str("{:.4f}".format(test_err)))
    # print("Gradient with bias Stupid example solution:" + str(T))

    # test using numpy
    T, err, _, _ = np.linalg.lstsq(X, y, rcond=None)
    print("Stupid example numpy err:" + str(err))
    print("Stupid example numpy solution:" + str(T))

    # test sklearn
    lr = LinearRegression()
    lr.fit(X, y)
    T = lr.coef_[0]
    print("Stupid example sklearn solution:" + str(T))
    print("--------")


# https://towardsdatascience.com/least-squares-linear-regression-in-python-54b87fc49e77
def another_example():
    print("Runing second example")

    X = np.array([[1, 2],
                  [1, 3],
                  [1, 4],
                  [1, 5],
                  [1, 6],
                  [1, 7],
                  [1, 8]])
    y = np.array([[5],
                  [4],
                  [3],
                  [4],
                  [6],
                  [7],
                  [9]])

    # Torch analytical version
    print("Running LST Torch analytical solution")
    mse = LeastSquareTransformation(method='analytical_torch', disable_shape_check=True)
    T = mse.compute_T_mat(X, y)
    assert T[0][0] == approx(1.678, abs=1e-3)
    assert T[1][0] == approx(0.75, abs=1e-3)

    test_err = mse.compute_error(X, y, T)
    print("Torch analytical ver2 test error:" + str("{:.4f}".format(test_err)))
    print("Torch analytical ver2 solution:" + str(T))

    # Torch SGD ver2
    mse = LeastSquareTransformation(lr=0.001, method='sgd_torch_2', disable_shape_check=True)
    T = mse.compute_T_mat(X, y)
    assert T[0][0] == approx(1.678, abs=1e-3)
    assert T[1][0] == approx(0.75, abs=1e-3)

    test_err = mse.compute_error(X, y, T)
    print("Torch SGD ver2 test error:" + str("{:.4f}".format(test_err)))
    print("Torch SGD ver2 solution:" + str(T))

    # Torch sgd
    mse = LeastSquareTransformation(lr=0.001, method='sgd_torch', disable_shape_check=True)
    T = mse.compute_T_mat(X, y)

    assert T[0][0] == approx(1.678, abs=1e-3)
    assert T[1][0] == approx(0.75, abs=1e-3)

    test_err = mse.compute_error(X, y, T)
    print("Torch SGD test error:" + str("{:.4f}".format(test_err)))
    print("Torch SGD solution:" + str(T))


    # Analytical
    mse = LeastSquareTransformation(verbose=False, disable_shape_check=True)
    T = mse.compute_T_mat(X, y)
    assert T[0][0] == approx(1.678, abs=1e-3)
    assert T[1][0] == approx(0.75, abs=1e-3)

    test_err = mse.compute_error(X, y, T)
    print("Analytical test error:" + str("{:.4f}".format(test_err)))
    print("Analytical solution:" + str(T))

    # # Gradient without bias
    # mse = LeastSquareTransformation(method='sgd', use_bias=False, verbose=False, disable_shape_check=True)
    # T = mse.compute_T_mat(X, y)
    # assert T[0][0] == approx(1.678, abs=1e-3)
    # assert T[1][0] == approx(0.75, abs=1e-3)
    #
    # test_err = mse.compute_error(X, y, T)
    # print("Gradient without bias test error:" + str("{:.4f}".format(test_err)))
    # print("Gradient without bias solution:" + str(T))
    #
    # # Gradient without bias, regularization = 0.0
    # mse = LeastSquareTransformation(method='sgd', use_bias=False, verbose=False, regularization=0.0,
    #                                 disable_shape_check=True)
    # T = mse.compute_T_mat(X, y)
    # assert T[0][0] == approx(1.678, abs=1e-3)
    # assert T[1][0] == approx(0.75, abs=1e-3)
    #
    # test_err = mse.compute_error(X, y, T)
    # print("Gradient without bias, regularization = 0.0 test error:" + str("{:.4f}".format(test_err)))
    # print("Gradient without bias, regularization = 0.0 solution:" + str(T))
    #
    # # Gradient without bias, regularization = 0.0001
    # mse = LeastSquareTransformation(method='sgd', use_bias=False, verbose=False, regularization=0.0001,
    #                                 disable_shape_check=True)
    # T = mse.compute_T_mat(X, y)
    # assert T[0][0] == approx(1.678, abs=1e-3)
    # assert T[1][0] == approx(0.75, abs=1e-3)
    #
    # test_err = mse.compute_error(X, y, T)
    # print("Gradient without bias, regularization = 0.0001 test error:" + str("{:.4f}".format(test_err)))
    # print("Gradient without bias, regularization = 0.0001 solution:" + str(T))
    #
    # # Gradient with bias
    # mse = LeastSquareTransformation(method='sgd', use_bias=True, verbose=False, disable_shape_check=True)
    # T = mse.compute_T_mat(X, y)
    # # assert T[0][0] == approx(1.678, abs=1e-3)
    # assert T[2][0] == approx(0.75, abs=1e-3)
    #
    # test_err = mse.compute_error(X, y, T)
    # print("Gradient with bias Test error:" + str(test_err))
    # print("Gradient with bias Test solution:" + str(T))
    #
    # # Gradient with bias, regularization = 0.0
    # mse = LeastSquareTransformation(method='sgd', use_bias=True, verbose=False, regularization=0.0,
    #                                 disable_shape_check=True)
    # T = mse.compute_T_mat(X, y)
    # # assert T[0][0] == approx(1.678, abs=1e-3)
    # assert T[2][0] == approx(0.75, abs=1e-3)
    #
    # test_err = mse.compute_error(X, y, T)
    # print("Gradient with bias, regularization = 0.0 Test error:" + str(test_err))
    # print("Gradient with bias, regularization = 0.0 Test solution:" + str(T))
    #
    # # Gradient with bias, regularization = 0.0001
    # mse = LeastSquareTransformation(method='sgd', use_bias=True, verbose=False, regularization=0.001,
    #                                 disable_shape_check=True)
    # T = mse.compute_T_mat(X, y)
    # # assert T[0][0] == approx(1.678, abs=1e-3)
    # assert T[2][0] == approx(0.75, abs=1e-3)
    #
    # test_err = mse.compute_error(X, y, T)
    # print("Gradient with bias, regularization = 0.0001 Test error:" + str(test_err))
    # print("Gradient with bias, regularization = 0.0001 Test solution:" + str(T))

    # test using numpy
    T, err, _, _ = np.linalg.lstsq(X, y, rcond=None)
    print("Test numpy err:" + str(err))
    print("Test numpy solution:" + str(T))

    # test sklearn
    lr = LinearRegression()
    lr.fit(X, y)
    T = lr.coef_
    # pak je jeste neco v intercept
    print("Test sklearn solution:" + str(T))
    print("--------")


def cross_lingual_example():
    print("Running Cross lingual example")

    # test emb
    pairs_path = os.path.join(LINKS_DIR, 'en-monolingual.txt')
    src_emb, trg_emb = load_word_vectors(
        os.path.join(EMB_TEST, 'w2v.english_corpus1.75_window-5_iter-10_min-count-5.vec'),
        os.path.join(EMB_TEST, 'w2v.english_corpus2.75_window-5_iter-10_min-count-5.vec'))

    src_emb.vectors = normalize(src_emb.vectors)
    trg_emb.vectors = normalize(trg_emb.vectors)

    src_word2id = {}
    for word in src_emb.vocab.keys():
        vocab = src_emb.vocab[word]
        src_word2id[word] = vocab.index

    trg_word2id = {}
    for word in trg_emb.vocab.keys():
        vocab = trg_emb.vocab[word]
        trg_word2id[word] = vocab.index

    X, Y, _ = build_transform_vectors(src_emb.vectors, trg_emb.vectors, 'list', src_word2id, trg_word2id, pairs_path)

    # Analytical
    mse = LeastSquareTransformation(use_bias=False)
    T_analytical = mse.compute_T_mat(X, Y)
    err_analytical = mse.compute_error(X, Y, T_analytical)
    print("Analytical err:" + str(err_analytical))

    # Analytical Regu
    mse = LeastSquareTransformation(use_bias=False, regularization=1000)
    T_analytical_2 = mse.compute_T_mat(X, Y)
    err_analytical_2 = mse.compute_error(X, Y, T_analytical_2)

    # # Gradient without bias
    # mse = LeastSquareTransformation(gradient_descent=True, use_bias=False)
    # T_gradient_without_bias = mse.compute_T_mat(X, Y)
    # err_gradient_without_bias = mse.compute_error(X, Y, T_gradient_without_bias)
    #
    # # Gradient without bias, reguralization=0.0001
    # mse = LeastSquareTransformation(gradient_descent=True, use_bias=False, regularization=0.0001, n_iters=100000)
    # T_gradient_with_bias = mse.compute_T_mat(X, Y)
    # err_gradient_without_bias_reguralized = mse.compute_error(X, Y, T_gradient_with_bias)
    #
    # # Gradient with bias
    # mse = LeastSquareTransformation(gradient_descent=True, use_bias=True)
    # T_gradient_with_bias = mse.compute_T_mat(X, Y)
    # err_gradient_with_bias = mse.compute_error(X, Y, T_gradient_with_bias)
    #
    # # Gradient with bias, reguralization=0.0001
    # mse = LeastSquareTransformation(gradient_descent=True, use_bias=True, regularization=0.0001, n_iters=100000)
    # T_gradient_with_bias = mse.compute_T_mat(X, Y)
    # err_gradient_with_bias_reguralized = mse.compute_error(X, Y, T_gradient_with_bias)

    # test using numpy
    T, err, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    print("Numpy err:" + str(np.sum(err)))
    print("Analytical err:" + str(err_analytical))
    print("Analytical reg err:" + str(err_analytical_2))
    # print("Without bias  : " + str(err_gradient_without_bias))
    # print("Without bias reguralization  0.0001 : " + str(err_gradient_without_bias_reguralized))
    # print("With bias     :" + str(err_gradient_with_bias))
    # print("With bias  eguralization  0.0001   :" + str(err_gradient_with_bias_reguralized))
    # mse.transform(X, Y)


def mainLSTtest():
    # test
    np.set_printoptions(precision=3, suppress=True)
    stupid_example_test()
    another_example()



    print("Running LST Analytical Torch Cross-lingual")
    lst = LeastSquareTransformation(method='analytical_torch', disable_shape_check=True)
    default_czech_english_test(lst)
    print("*" * 70)

    # pokud to normuju (jednotkove vektory) pres vektory slov  norm_unit_feature=False tak musi byt learning rate male,
    # pokud to normuju (jednotkove vektory) pres featury norm_unit_feature=False tak learning rate nemusi byt tak mala

    lst = LeastSquareTransformation(lr=0.0001, method='sgd_torch_2', disable_shape_check=True, batch_size=100000000, n_iters=5000)
    default_czech_english_test(lst, norm_unit_feature=False)
    print("*" * 70)

    print("Running LST SGD Torch Cross-lingual")
    # Torch sgd
    lst = LeastSquareTransformation(lr=0.0001, method='sgd_torch', disable_shape_check=True, batch_size=1000000, n_iters=5000)
    default_czech_english_test(lst, norm_unit_feature=False)
    print("*" * 70)





    print("Running true least square cross ling analytical with regu: 0.1")
    lst = LeastSquareTransformation(regularization=0.1)
    default_czech_english_test(lst)

    print("*" * 70)

    print("Running true least square cross ling analytical with regu: 0.0")
    lst = LeastSquareTransformation(regularization=0.0)
    default_czech_english_test(lst)

    cross_lingual_example()

    # print("*" * 70)
    #
    # print("Running true least square cross ling Gradient desc with regu: 0.0")
    # lst = LeastSquareTransformation(regularization=0.0, gradient_descent=True)
    # default_czech_english_test(lst)
    #
    # print("*" * 70)
    #
    # print("Running true least square cross ling Gradient desc with regu: 0.1")
    # lst = LeastSquareTransformation(regularization=0.1, gradient_descent=True)
    # default_czech_english_test(lst)
