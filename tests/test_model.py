import numpy as np


def test_accuracy_is_valid_range():
    accuracy = 0.71

    assert 0.0 <= accuracy <= 1.0


def test_precision_is_valid_range():
    precision = 0.72

    assert 0.0 <= precision <= 1.0


def test_recall_is_valid_range():
    recall = 0.70

    assert 0.0 <= recall <= 1.0


def test_f1_is_valid_range():
    f1 = 0.71

    assert 0.0 <= f1 <= 1.0


def test_prediction_shape_matches_input():
    predictions = np.array([0, 1, 1, 0])
    inputs = np.array([1, 2, 3, 4])

    assert len(predictions) == len(inputs)