import numpy as np

from OrthogonalRankingTransformation import OrthogonalRankingTransformation
from tests.CrossLingualTest import default_czech_english_test


def mainORTtest():
    np.set_printoptions(precision=3, suppress=True)

    ranking_transformation = OrthogonalRankingTransformation(disable_shape_check=True)
    default_czech_english_test(ranking_transformation, norm_unit_feature=True)
