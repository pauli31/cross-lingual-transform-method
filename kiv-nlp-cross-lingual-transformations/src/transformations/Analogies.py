from OrthogonalRankingTransformation import OrthogonalRankingTransformation
from tests.AnalogiesTest import analogies_test

if __name__ == '__main__':

    # lowercase by mel byt true pro nase embeddingy

    # transformation = OrthogonalTransformation()
    # transformation = LeastSquareTransformation()
    # transformation = CanonicalCorrelationAnalysis()
    # transformation = RankingTransformation()
    transformation = OrthogonalRankingTransformation()
    # Puvodni parametry se kterejma sem to poustel poprve, pak je adam menil ty defaultni
    load_additional = True
    custom_embeddings = False
    lowercase = True

    # ---------------
    normalize_after_transformation = True
    normalize_before = False
    # ---------------

    # analogies_test(src_lng='cs', trg_lng='en', transformation=transformation, lowercase=lowercase, normalize_after_transformation=normalize_after_transformation, normalize_before=normalize_before, custom_embeddings=custom_embeddings, load_additional=load_additional)
    analogies_test(src_lng='en', trg_lng='cs', transformation=transformation, lowercase=lowercase, normalize_after_transformation=normalize_after_transformation, normalize_before=normalize_before, custom_embeddings=custom_embeddings, load_additional=load_additional)

    print("load_additional:" + str(load_additional))
    print("custom_embeddings:" + str(custom_embeddings))
    print("lowercase:" + str(lowercase))
    print("normalize_after_transformation:" + str(normalize_after_transformation))
    print("normalize_before:" + str(normalize_before))

    print('%' * 70)
    print('%' * 70)
    # print("custom_embeddings:" + str(custom_embeddings))
    # print("lowercase:" + str(lowercase))
    print("normalize_after_transformation:" + str(normalize_after_transformation))
    print("normalize_before:" + str(normalize_before))