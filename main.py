"""
Source code to reproduce experiments of Herron 2012 paper.
"""
import numpy as np
import matplotlib.pyplot as plt


def create_manuscripts(number_of_manuscripts, seed=None):
    """
    Generate a hypothetical manuscript which
    corresponds to a number uniformly sampled from 1 - 10.
    """
    np.random.seed(seed)
    return 10 * np.random.random(size=number_of_manuscripts)


def review_manuscripts(manuscripts, imprecision_error_sd, other_error_sd):
    """
    Given a collection of manuscripts, add the imprecision error
    and the other error as sampled from a normal distribution
    """
    number_of_manuscripts = len(manuscripts)
    imprecision_error = np.random.normal(
        scale=imprecision_error_sd, size=number_of_manuscripts
    )
    other_error = np.random.normal(scale=other_error_sd, size=number_of_manuscripts)
    return np.clip(a=manuscripts + imprecision_error + other_error, a_min=0, a_max=10)


def is_above_threshold(
    manuscripts,
    threshold,
):
    """
    Return whether or not the manuscrupts pass the threshold
    """
    return manuscripts >= threshold


def get_average_review_scores(
    manuscripts,
    number_of_reviews,
    imprecision_error_sd,
    other_error_sd,
):
    average_review_scores = np.mean(
        [
            review_manuscripts(
                manuscripts=manuscripts,
                imprecision_error_sd=imprecision_error_sd,
                other_error_sd=other_error_sd,
            )
            for _ in range(number_of_reviews)
        ],
        axis=0,
    )
    return average_review_scores


def is_above_threshold_based_on_average(
    manuscripts,
    number_of_reviews,
    threshold,
    imprecision_error_sd,
    other_error_sd,
):
    """
    Repeat the reviews and return booleans on
    wether or not to accept based on average score.
    """
    average_review_scores = np.mean(
        [
            review_manuscripts(
                manuscripts=manuscripts,
                imprecision_error_sd=imprecision_error_sd,
                other_error_sd=other_error_sd,
            )
            for _ in range(number_of_reviews)
        ],
        axis=0,
    )
    return average_review_scores >= threshold


def is_above_threshold_based_on_vote(
    manuscripts,
    number_of_reviews,
    threshold,
    imprecision_error_sd,
    other_error_sd,
):
    """
    Repeat the reviews and return booleans on
    wether or not to accept based on voting.
    """
    number_of_votes = np.sum(
        [
            review_manuscripts(
                manuscripts=manuscripts,
                imprecision_error_sd=imprecision_error_sd,
                other_error_sd=other_error_sd,
            )
            >= threshold
            for _ in range(number_of_reviews)
        ],
        axis=0,
    )
    return number_of_votes >= number_of_reviews / 2


def accuracy_of_process(
    manuscripts,
    threshold,
    process,
    number_of_reviews,
    imprecision_error_sd,
    other_error_sd,
):
    accurate_decisions = is_above_threshold(
        manuscripts=manuscripts,
        threshold=threshold,
    )
    decisions = process(
        manuscripts=manuscripts,
        threshold=threshold,
        number_of_reviews=number_of_reviews,
        imprecision_error_sd=imprecision_error_sd,
        other_error_sd=other_error_sd,
    )
    return np.sum(decisions == accurate_decisions) / len(manuscripts)
