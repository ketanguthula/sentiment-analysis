import pandas as pd
import pytest

from src.preprocess import clean_text, clean_text_series, build_vectorizer


def test_clean_text_lowercases_text():
    assert clean_text("I LOVE This") == "i love this"


def test_clean_text_removes_urls():
    assert "http" not in clean_text("Check this out http://example.com")


def test_clean_text_removes_mentions():
    assert "username" not in clean_text("@username this is great")


def test_clean_text_removes_special_characters():
    assert clean_text("Wow!!! This is #great :)") == "wow this is great"


def test_clean_text_rejects_non_string_input():
    with pytest.raises(TypeError):
        clean_text(123)


def test_clean_text_series_does_not_modify_original_series():
    series = pd.Series(["HELLO!!!"])
    original = series.copy()

    cleaned = clean_text_series(series)

    assert series.equals(original)
    assert cleaned.iloc[0] == "hello"


def test_build_vectorizer_creates_tfidf_vectorizer():
    vectorizer = build_vectorizer(max_features=100, ngram_range=(1, 2))

    assert vectorizer.max_features == 100
    assert vectorizer.ngram_range == (1, 2)