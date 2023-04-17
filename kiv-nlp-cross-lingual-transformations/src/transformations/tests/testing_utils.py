from gensim.models import KeyedVectors
from gensim.models import FastText as FT
import fasttext
import fasttext.util

def load_word_vectors(src_file_path, trg_file_path, max_vectors=None, binary=False, type='w2v'):
    if type == 'fasttext':
        # src_emb = FT.load_fasttext_format(src_file_path)
        # trg_emb = FT.load_fasttext_format(trg_file_path)
        src_emb = fasttext.load_model(src_file_path)
        trg_emb = fasttext.load_model(trg_file_path)
    elif type == 'w2v':
        src_emb = KeyedVectors.load_word2vec_format(src_file_path, binary=binary, limit=max_vectors)
        trg_emb = KeyedVectors.load_word2vec_format(trg_file_path, binary=binary, limit=max_vectors)
    else:
        raise Exception("Unknown embedding type: " + str(type))
    return src_emb, trg_emb