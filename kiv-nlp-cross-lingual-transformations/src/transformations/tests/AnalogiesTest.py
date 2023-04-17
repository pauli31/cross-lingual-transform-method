import logging
import os
import numpy as np
import uuid

from tests.config import ANALOGIES_DIR, EMB, LINKS_DIR
from CanonicalCorrelationAnalysis import CanonicalCorrelationAnalysis
from LeastSquareTransformation import LeastSquareTransformation
from OrthogonalRankingTransformation import OrthogonalRankingTransformation
from OrthogonalTransformation import OrthogonalTransformation
from RankingTransformation import RankingTransformation
from tests.testing_utils import load_word_vectors
from utils import build_transform_vectors, normalize, NormalizedFastText, \
    build_transform_vectors_fasttext, save_w2v_format_fasttext, load_additional_words

logger = logging.getLogger("CrossLingualTest")
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')


def analogies_test(src_lng, trg_lng, transformation, lowercase=False, normalize_after_transformation=False,
                   normalize_before=True, custom_embeddings=False,load_additional=False):
    """
    Test all analogy categories located inside analogies folder.
    Final score is average from all categories.

    :param src_lng: language of source embedding
    :param trg_lng: language of target embedding
    :param transformation: used transformation
    """
    src_emb, trg_emb = prepare_test_data(src_lng, trg_lng, transformation, normalize_after_transformation,
                                         normalize_before, custom_embeddings, load_additional)
    base_folder = ANALOGIES_DIR
    categories_list = os.listdir(base_folder)
    categories = len(categories_list)
    score = 0.0

    copy_paste_res = ''

    for category in categories_list:
        # Prepare analogies
        src_from, src_to = load_analogies(src_emb.vocab, os.path.join(base_folder, category, f"{src_lng}.txt"), lowercase=lowercase)
        trg_from, trg_to = load_analogies(trg_emb.vocab, os.path.join(base_folder, category, f"{trg_lng}.txt"), lowercase=lowercase)

        normalzied = normalize_before
        correct = 0
        norm_trg_vecs = np.linalg.norm(trg_emb.vectors, axis=1).reshape((trg_emb.vectors.shape[0], 1))
        # Loop through all analogies within current category
        for i in range(len(src_from)):
            # Computes similarities effectively using matrix multiplication (vectors must be normalized before!)
            if normalzied is True:
                similarities = (src_emb.vectors[src_to[i]] - src_emb.vectors[src_from[i]] + trg_emb.vectors[trg_from]) \
                               @ trg_emb.vectors.T
            else:
                tmp_vecs = (src_emb.vectors[src_to[i]] - src_emb.vectors[src_from[i]] + trg_emb.vectors[trg_from])
                norms_tmp_vecs = np.linalg.norm(tmp_vecs, axis=1).reshape((tmp_vecs.shape[0], 1))

                normalized_tmp_vecs = tmp_vecs/norms_tmp_vecs
                norm_vectors = trg_emb.vectors/norm_trg_vecs
                similarities = normalized_tmp_vecs @ norm_vectors.T


            # Make first target word dissimilar to the search word
            similarities[:, trg_from] = -1
            # Computed correctly found analogies
            correct += np.sum(np.argmax(similarities, axis=1) == trg_to)
        # Update score
        if correct == 0:
            category_score = 0
        else:
            category_score = (correct / float(len(src_to) * len(trg_to)))
        score += category_score
        logger.info(f"Analogies - {category}: {(category_score * 100):.4f} %")
        copy_paste_res += f"{(category_score * 100):.4f}\n"

    total_score = (score / float(categories) * 100)
    logger.info(70 * '-')
    logger.info(f"Final analogies accuracy: {total_score:.4f} %")
    logger.info(70 * '-')
    copy_paste_res += f"{total_score:.4f}\n"
    print(copy_paste_res)


def load_analogies(vectors, filename, delimiter='\t', lowercase=False):
    """
    Load one analogy category from given file.

    :param vectors: list of vectors in given semantic space
    :param filename: name of the file with analogies
    :param delimiter: separator of pairs in the file

    :return: two lists with indexes of analogies words
    """
    from_analogies, to_analogies = [], []

    with open(filename, encoding='utf-8') as file:
        for line in file:
            if len(line.rstrip()) == 0 or delimiter not in line:
                continue

            source, target = line.rstrip().split(delimiter)
            if lowercase is True:
                source = source.lower()
                target = target.lower()
            if source not in vectors or target not in vectors:
                continue
            from_analogies.append(vectors[source].index)
            to_analogies.append(vectors[target].index)

    return from_analogies, to_analogies


def prepare_test_data(src_lng, trg_lng, transformation, normalize_after_transformation=False,
                      normalize_before=True, custom_embeddings=False, load_additional=False):
    """
    Prepare data from analogies test. Load embeddings, dictionary and then transform source embedding
    with given transformation.

    :param src_lng: language of source embedding
    :param trg_lng: language of target embedding
    :param transformation: used transformation

    :return: transformed source embedding and target embedding
    """
    # Set up basic parameters
    if custom_embeddings is False:
        type = 'w2v'
        binary=False
        src_emb_file = os.path.join(EMB, f"cc.{src_lng}.300.vec")
        trg_emb_file = os.path.join(EMB, f"cc.{trg_lng}.300.vec")
    else:
        binary=True
        type = 'fasttext'
        src_emb_file = os.path.join(EMB, f"fasttext_{src_lng}.bin")
        trg_emb_file = os.path.join(EMB, f"fasttext_{trg_lng}.bin")

    max_vectors = 100000
    # links_path = os.path.join(LINKS_DIR, f"cs-en-50k.txt")
    # links_path = os.path.join(LINKS_DIR, f"{src_lng}-{trg_lng}-50k.txt")
    links_path = os.path.join(LINKS_DIR, f"{src_lng}_to_{trg_lng}_dic.csv")
    max_links = 20000

    # Print info
    logger.info(f"Testing cross-lingual analogies with Transformation: {transformation}")
    logger.info(f"Source language: {src_lng}")
    logger.info(f"Target language: {trg_lng}")
    logger.info(f"Source embedding file: {src_emb_file}")
    logger.info(f"Target embedding file: {trg_emb_file}")
    logger.info(f"Links file: {links_path}")
    logger.info(f"Maximal number of vectors: {max_vectors}")
    logger.info(f"Maximal number of links: {max_links}")

    # Load and normalize embeddings
    src_emb, trg_emb = load_word_vectors(src_emb_file, trg_emb_file, max_vectors, binary=binary, type=type)
    if type == 'fasttext':
        if load_additional is True:
            additional_src = load_additional_words(src_lng)
            additional_trg = load_additional_words(trg_lng)
        else:
            additional_src = None
            additional_trg = None

        random_src = str(uuid.uuid4())
        random_trg = str(uuid.uuid4())
        # normalize_before
        if normalize_before is True:
            src_emb = NormalizedFastText(src_emb, 300)
            trg_emb = NormalizedFastText(trg_emb, 300)

            src_emb_file_new = src_emb_file + "_" + random_src + "-w2v.vec"
            trg_emb_file_new = trg_emb_file + "_" + random_trg + "-w2v.vec"

        else:
            src_emb = NormalizedFastText(src_emb, 300, disable_normalization=True)
            trg_emb = NormalizedFastText(trg_emb, 300, disable_normalization=True)

            src_emb_file_new = src_emb_file + "_" + random_src + "-unormalized-w2v.vec"
            trg_emb_file_new = trg_emb_file + "_" + random_trg + "-unormalized-w2v.vec"

        logger.info("Saving new fasttext src into:" + str(src_emb_file_new))
        logger.info("Saving new fasttext trg into:" + str(trg_emb_file_new))

        save_w2v_format_fasttext(src_emb, src_emb_file_new, additional_src)
        save_w2v_format_fasttext(trg_emb, trg_emb_file_new, additional_trg)

        type = 'w2v'
        src_emb, trg_emb = load_word_vectors(src_emb_file_new, trg_emb_file_new, max_vectors, binary=False, type=type)

    elif type == 'w2v':
        if normalize_before is True:
            src_emb.vectors = normalize(src_emb.vectors)
            trg_emb.vectors = normalize(trg_emb.vectors)
        else:
            src_emb.vectors = src_emb.vectors
            trg_emb.vectors = trg_emb.vectors

    # Transform source embedding
    if type == 'fasttext':
        src_word2id = {word: src_emb.fasttext.get_word_id(word) for word in src_emb.fasttext.get_words() }
        trg_word2id = {word: trg_emb.fasttext.get_word_id(word) for word in trg_emb.fasttext.get_words() }
        X, Y, final_list = build_transform_vectors_fasttext(src_emb, trg_emb, 'list', src_word2id, trg_word2id,
                                                   links_path, max=max_links)
    elif type == 'w2v':
        src_word2id = {word: src_emb.vocab[word].index for word in src_emb.vocab.keys()}
        trg_word2id = {word: trg_emb.vocab[word].index for word in trg_emb.vocab.keys()}
        X, Y, final_list = build_transform_vectors(src_emb.vectors, trg_emb.vectors, 'list', src_word2id, trg_word2id,
                                                   links_path, max=max_links)
    T_matrix = transformation.compute_T_mat(X, Y)
    src_emb.vectors = transformation.transform_with_T(src_emb.vectors, T_matrix)

    if normalize_after_transformation is True:
        src_emb.vectors = normalize(src_emb.vectors)

    return src_emb, trg_emb




