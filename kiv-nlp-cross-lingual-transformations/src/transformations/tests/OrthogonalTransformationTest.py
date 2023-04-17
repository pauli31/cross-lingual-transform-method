import os

import numpy as np
from pytest import approx

from tests.config import EMB_TEST, LINKS_DIR
from OrthogonalTransformation import OrthogonalTransformation
from tests.CrossLingualTest import default_czech_english_test
from tests.testing_utils import load_word_vectors
from utils import normalize, build_transform_vectors, compute_cosine


def stupid_example_test():
    print("Running stupid example")

    # stupid linear test
    X = np.array([[1, 1],
                  [1, 2],
                  [1, 3]])
    y = np.array([[1],
                  [2],
                  [3]])

    ot = OrthogonalTransformation()
    T = ot.compute_T_mat(X, y)
    assert T[0][0] == approx(0, abs=1e-3)
    assert T[1][0] == approx(1, abs=1e-3)

    pass


def cross_lingual_example(method):
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

    ot = OrthogonalTransformation(method=method)
    T = ot.compute_T_mat(X, Y)
    err = ot.compute_error(X, Y, T)
    print("Analytical err:" + str(err))

    X_transformed = ot.transform(X, Y)

    rand_x = [654, 3256, 8756, 1500, 6584]
    rand_y = [5654, 5481, 987, 8486, 13]
    for i in range(0, len(rand_x)):
        x_ind = rand_x[i]
        y_ind = rand_y[i]

        x_orig = X[x_ind]
        x_orig_y_ind = X[y_ind]
        x_trans = X_transformed[x_ind]
        x_trans_y_ind = X_transformed[y_ind]
        y = Y[y_ind]

        # i centered everything so it will be 1
        # x_orig_len = compute_vec_len(x_orig)
        # x_trans_len = compute_vec_len(x_trans)

        x_orig_x_trans_cosine = compute_cosine(x_orig, x_trans)

        x_orig_y_cosine = compute_cosine(x_orig, y)
        x_trans_y_cosine = compute_cosine(x_trans, y)

        # cosine between two vectors from one space, the angle has to be the same
        x_orig_x_cosine = compute_cosine(x_orig, x_orig_y_ind)
        x_trans_x_cosine = compute_cosine(x_trans, x_trans_y_ind)

        # tohle by nemelo byt podobne, podobnost mezi transformovanym  vektorem a originalnim
        print("Cosine between trans and orig:" + str(x_orig_x_trans_cosine))

        # Tohle by tasi taky nemelo byt podobne, pac beru podobnost mezi transformovanym
        # a nahodnym vektorem y
        print("Cosine between orig and y:" + str(x_orig_y_cosine))
        # tohle taky nemusi byt podobne
        print("Cosine between trans and y:" + str(x_trans_y_cosine))
        print("##")

        # tyhle by meli byt podobne pac by to melo zachovavat uhly, tj. uhly ve zdrojovym
        # by meli byt stejny jako uhly zdrojoveho po transformaci
        print("Cosine between orig x vectors:" + str(x_orig_x_cosine))
        print("Cosine between trans x vectors:" + str(x_trans_x_cosine))
        assert x_orig_x_cosine == approx(x_trans_x_cosine, abs=1e-5)
        print("----")

    print("---------------------------")
    print("---------------------------")
    # nefunguje pac tam posilam jenom matice pro transformaci, kde jsou zprehazeny
    # indexy oproti tomu puvodnimu prostoru
    # prvni je laugh, a jeho indexy
    words = ["laugh"]
    words_id_x = [654]
    words_id_y = [628]

    for i in range(0, len(words_id_x)):
        x_ind = words_id_x[i]
        y_ind = words_id_y[i]

        x_trans = X_transformed[x_ind]
        y = Y[y_ind]

        cosine = compute_cosine(x_trans, y)
        print("Cosine for word:" + str(words[i]) + " is:" + str(cosine))


def mainOTTest():
    # stupid_example_test()

    print("Running Orthogonal transformation torch")
    ot = OrthogonalTransformation(method='torch')
    default_czech_english_test(ot, norm_unit_feature=False)

    print("*" * 70)
    cross_lingual_example('torch')

    # print("Running Orthogonal transformation numpy")
    # ot = OrthogonalTransformation()
    # default_czech_english_test(ot, norm_unit_feature=False)
    #
    # print("*" * 70)
    #
    # cross_lingual_example('numpy')
