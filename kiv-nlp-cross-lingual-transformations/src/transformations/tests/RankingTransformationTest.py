import numpy as np

from src.transformations.RankingTransformation import RankingTransformation
from src.transformations.tests.CrossLingualTest import default_czech_english_test


def mainRTtest():
    np.set_printoptions(precision=3, suppress=True)

    ranking_transformation = RankingTransformation(disable_shape_check=True)
    default_czech_english_test(ranking_transformation, norm_unit_feature=True)
