from tests.config import EMB, LINKS_DIR
from OrthogonalTransformation import OrthogonalTransformation
import os
import logging
import numpy as np

from tests.testing_utils import load_word_vectors
from utils import normalize, build_transform_vectors, compute_cosine

logger = logging.getLogger("CrossLingualTest")
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')


def default_czech_english_test(transformation, assert_test=True, norm_unit_feature=False):
    """
    Performs default cross-lingual test between Czech and English fasttext emeddings,
    with maximum 100 000 vectors and maximum 20 000 links, with default list of pairs

    :param transformation: the transformation to be used an instance of class Transform
    :param assert_test: whether enable the assert test, i.e., if the target word in list of word pairs
                        must be in top-10 nearest nearest neighbors
    """
    # transformation = OrthogonalTransformation()
    # transformation = LeastSquareTransformation(regularization=0.01)
    cs_src = os.path.join(EMB, 'cc.cs.300.vec')
    en_trg = os.path.join(EMB, 'cc.en.300.vec')
    max_vectors = 100000
    links_path = os.path.join(LINKS_DIR, 'cs-en-50k.txt')
    # links_path = os.path.join(LINKS_DIR, 'cs_to_en_dic-jakub.csv')
    max_links = 20000
    test_pairs = [('auto', 'car'), ('král', 'king'), ('a', 'and'), ('Londýn', 'London'), ('kočka', 'cat'),
                  ('jablko', 'apple')]
    cross_lingual_test(transformation, cs_src, en_trg, max_vectors, links_path, max_links, test_pairs, assert_test,
                       norm_unit_feature=norm_unit_feature)


def cross_lingual_test(transformation, src_emb_file, trg_emb_file, max_vectors, links, max_links, test_pairs,
                       assert_test=True, norm_unit_feature=False):
    """
    Performs cross-lingual between given word embeddings with a given transformation method

    :param transformation: the transformation to be used an instance of class Transform
    :param src_emb_file: path to .vec file with source embeddings
    :param trg_emb_file: path to .vec file with target embeddings
    :param max_vectors: maximum vectors that will be loaded
    :param links: list of tuples (pairs of achnor words for transformation) or path to file with these pairs
    :param max_links: maximum number of links that will be used for the transformation, i.e., first n words
    :param test_pairs: list of tuples (pairs) of words that will be tested
    :param assert_test: if set to True, in 10 nearest neighbors of transformed vector for source word
                        there must be the target word
                        if set to False, no check is performed

    """

    logger.info('Evaluation crossilngual with Transformation:' + str(transformation))
    logger.info('src emb file:' + src_emb_file)
    logger.info('trg emb file:' + trg_emb_file)
    logger.info('max vectors:' + str(max_vectors))
    logger.info('links:' + str(links))
    logger.info('max links:' + str(max_links))
    logger.info('test pairs:' + str(test_pairs))

    src_emb, trg_emb = load_word_vectors(src_emb_file, trg_emb_file, max_vectors)

    src_emb.vectors = normalize(src_emb.vectors, unit_vec_features=norm_unit_feature)
    trg_emb.vectors = normalize(trg_emb.vectors, unit_vec_features=norm_unit_feature)

    src_word2id = {}
    for word in src_emb.vocab.keys():
        vocab = src_emb.vocab[word]
        src_word2id[word] = vocab.index

    trg_word2id = {}
    for word in trg_emb.vocab.keys():
        vocab = trg_emb.vocab[word]
        trg_word2id[word] = vocab.index

    X, Y, final_list = build_transform_vectors(src_emb.vectors, trg_emb.vectors, 'list', src_word2id, trg_word2id, links, max=max_links)

    T_matrix = transformation.compute_T_mat(X, Y)
    err = transformation.compute_error(X, Y, T_matrix)
    logger.info("Error on compute error:" + str(err))
    src_emb_transformed = transformation.transform_with_T(src_emb.vectors, T_matrix)

    # trg_emb.similar_by_vector(src_emb_transformed[src_word2id['a']])
    # compute_cosine(src_emb_transformed[src_word2id['auto']], src_emb_transformed[src_word2id['vozidlo']])

    avg_cosine = []

    for (src_word, trg_word) in test_pairs:
        if not src_word in src_word2id:
            logger.info("Source word:" + str(src_word) + " is not in a vocabulary")

        if not trg_word in trg_word2id:
            logger.info("Target word:" + str(src_word) + " is not in a vocabulary")

        src_idx = src_word2id[src_word]
        trg_idx = trg_word2id[trg_word]

        src_vec = src_emb_transformed[src_idx]
        trg_vec = trg_emb.vectors[trg_idx]

        cosine = compute_cosine(src_vec, trg_vec)
        avg_cosine.append(cosine)

        logger.info(70* '-')
        logger.info("Cosine between words {} : {} is {:.4f}".format(src_word, trg_word, cosine))
        logger.info(70 * '-')
        logger.info("Most similar to transformed vector for word:" + str(src_word))
        most_similar_to_src_vec = trg_emb.similar_by_vector(src_vec)

        if assert_test is True:
            # check that the target word is in top n nearest neighbors
            tmp_words = [word_tuple[0] for word_tuple in most_similar_to_src_vec]
            # assert trg_word in tmp_words

            for (word, sim) in most_similar_to_src_vec:
                logger.info(str(word) + " : {:.4f}".format(sim))
            logger.info(70 * '#')

    avg_cosine = np.mean(avg_cosine)
    logger.info(70 * '%')
    logger.info("Average cosine is:{:.4f}".format(avg_cosine))
    logger.info(70 * '%')


if __name__ == '__main__':
    transformation = OrthogonalTransformation()
    # transformation = LeastSquareTransformation()
    # transformation = LeastSquareTransformation(regularization=0.1)
    # transformation = LeastSquareTransformation(gradient_descent=True, regularization=0.1)
    # transformation = CanonicalCorrelationAnalysis()
    default_czech_english_test(transformation)
    print('%'*70)
    print('%'*70)

    # transformation = OrthogonalTransformation()
    # default_czech_english_test(transformation)
