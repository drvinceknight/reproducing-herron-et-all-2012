"""
Source code to reproduce experiments of Herron 2012 paper.
"""
import numpy as np


def create_manuscripts(number_of_manuscripts, seed=None):
    """
    Generate a hypothetical manuscript which
    corresponds to a number uniformly sampled from 1 - 10.

    Args:
        number_of_manuscripts : int
            The number of manuscripts to be sampled
        seed : int
            The random seed for the number generator. Allows for reproducible sampling.

    Returns:
        array
        A numpy array of integers corresponding to the manuscripts.
    """
    np.random.seed(seed)
    return np.random.randint(low=1, high=10 + 1, size=number_of_manuscripts)


def review_manuscripts(manuscripts, imprecision_error_sd, other_error_sd):
    """
    Given a collection of manuscripts, add the imprecision error
    and the other error as sampled from a normal distribution

    Args:
        manuscripts : array
            A numpy array of integers corresponding to manuscripts.
        imprecision_error_sd : float
            The standard deviation corresponding to the imprecision error
        other_error_sd : float
            The standard deviation corresponding to the other error

    Returns:
        array
        A numpy array of floats corresponding to the reviews of each manuscript.
    """
    number_of_manuscripts = len(manuscripts)
    imprecision_error = np.random.normal(
        scale=imprecision_error_sd, size=number_of_manuscripts
    )
    other_error = np.random.normal(scale=other_error_sd, size=number_of_manuscripts)
    return np.clip(a=manuscripts + imprecision_error + other_error, a_min=1, a_max=10)


def is_above_threshold(
    manuscripts,
    threshold,
):
    """
    Return whether or not the manuscrupts pass the threshold

    Args:
        manuscripts : array
            A numpy array of integers corresponding to manuscripts.
        threshold : int
            A numeric score corresponding to an acceptance threshold for a manuscript.

    Returns:
        array
        A numpy array of booleans.
    """
    return manuscripts >= threshold


def get_average_review_scores(
    manuscripts,
    number_of_reviews,
    imprecision_error_sd,
    other_error_sd,
):
    """
    Get average review scores over a number of reviewers.

    Args:
        manuscripts : array
            A numpy array of integers corresponding to manuscripts.
        number_of_reviews : int
            The number of time to repeat the review process.
        imprecision_error_sd : float
            The standard deviation corresponding to the imprecision error
        other_error_sd : float
            The standard deviation corresponding to the other error

    Returns:
        array
        A numpy array of floats corresponding to the average review of each manuscript.
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

    Args:
        manuscripts : array
            A numpy array of integers corresponding to manuscripts.
        number_of_reviews : int
            The number of time to repeat the review process.
        threshold : int
            A numeric score corresponding to an acceptance threshold for a manuscript.
        imprecision_error_sd : float
            The standard deviation corresponding to the imprecision error
        other_error_sd : float
            The standard deviation corresponding to the other error

    Returns:
        array
        A numpy array of booleans.
    """
    average_review_scores = get_average_review_scores(
        manuscripts=manuscripts,
        number_of_reviews=number_of_reviews,
        imprecision_error_sd=imprecision_error_sd,
        other_error_sd=other_error_sd,
    )
    return average_review_scores >= threshold


def count_votes(reviews, threshold):
    """
    Given an iterable of iterables of reviews returns the number of votes for each paper.


    Args:
        reviews : array
            A numpy array of floats corresponding to the average review of each manuscript.
        threshold : int
            A numeric score corresponding to an acceptance threshold for a manuscript.

    Returns:
        array
        A numpy array of integers corresponding to the number of votes in favour of accepting a
        paper.
    """
    number_of_votes = np.sum(
        [review >= threshold for review in reviews],
        axis=0,
    )
    return number_of_votes


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

    Args:
        manuscripts : array
            A numpy array of integers corresponding to manuscripts.
        number_of_reviews : int
            The number of time to repeat the review process.
        threshold : int
            A numeric score corresponding to an acceptance threshold for a manuscript.
        imprecision_error_sd : float
            The standard deviation corresponding to the imprecision error
        other_error_sd : float
            The standard deviation corresponding to the other error

    Returns:
        array
        A numpy array of booleans indicating if the number of votes for a given paper
        is above the majority.
    """
    reviews = (
        review_manuscripts(
            manuscripts=manuscripts,
            imprecision_error_sd=imprecision_error_sd,
            other_error_sd=other_error_sd,
        )
        for _ in range(number_of_reviews)
    )
    number_of_votes = count_votes(reviews=reviews, threshold=threshold)
    return number_of_votes >= int(number_of_reviews / 2)


def accuracy_of_process(
    manuscripts,
    threshold,
    process,
    number_of_reviews,
    imprecision_error_sd,
    other_error_sd,
):
    """
    Return the accuracy of a given review process.

    Repeat the reviews and return booleans on
    wether or not to accept based on voting.

    Args:
        manuscripts : array
            A numpy array of integers corresponding to manuscripts.
        threshold : int
            A numeric score corresponding to an acceptance threshold for a manuscript.
        process : function
            A review function
        number_of_reviews : int
            The number of time to repeat the review process.
        imprecision_error_sd : float
            The standard deviation corresponding to the imprecision error
        other_error_sd : float
            The standard deviation corresponding to the other error

    Returns:
        float
        The proportion of papers that were accepted correctly or rejected correctly.
    """
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
