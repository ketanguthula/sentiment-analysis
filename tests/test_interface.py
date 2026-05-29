import pytest

from src.app import validate_user_input, predict_sentiment


class DummyModel:
    def predict(self, texts):
        return [1]

    def predict_proba(self, texts):
        return [[0.2, 0.8]]


def test_validate_user_input_accepts_valid_text():
    is_valid, error_message = validate_user_input(
        "I really love this new update."
    )

    assert is_valid is True
    assert error_message is None


def test_validate_user_input_rejects_empty_text():
    is_valid, error_message = validate_user_input("")

    assert is_valid is False
    assert "Please enter" in error_message


def test_validate_user_input_rejects_too_short_text():
    is_valid, error_message = validate_user_input("Bad")

    assert is_valid is False
    assert "too short" in error_message


def test_predict_sentiment_returns_expected_fields():
    model = DummyModel()

    result = predict_sentiment(model, "I love this product")

    assert result["label"] == "positive"
    assert result["prediction"] == 1
    assert result["confidence"] == 0.8