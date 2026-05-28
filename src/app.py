import os
import sys
from pathlib import Path

import joblib
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "models/best_model.joblib"


def validate_user_input(text: str):
    if not isinstance(text, str) or not text.strip():
        return False, "Please enter a tweet, review, comment, or short text to analyze."

    if len(text.strip().split()) < 3:
        return False, "The input is too short to analyze reliably. Please provide a longer sentence."

    return True, None


def load_model(model_path=MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python src/train.py` first."
        )

    return joblib.load(model_path)


def predict_sentiment(model, text: str):
    prediction = model.predict([text])[0]

    label = "positive" if prediction == 1 else "negative"

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = float(max(probabilities))

    return {
        "label": label,
        "prediction": int(prediction),
        "confidence": confidence,
    }


def generate_fallback_explanation(text: str, prediction_result: dict):
    label = prediction_result["label"]
    confidence = prediction_result["confidence"]

    if confidence is not None:
        confidence_text = f" with about {confidence:.1%} confidence"
    else:
        confidence_text = ""

    return (
        f"The trained sentiment model classified this text as **{label}**"
        f"{confidence_text}. This prediction is based on patterns learned from "
        f"labeled tweets in the Sentiment140 dataset. Because tweets can contain "
        f"sarcasm, slang, and missing context, this result should be interpreted "
        f"as a model estimate rather than a guaranteed judgment."
    )


def generate_llm_explanation(text: str, prediction_result: dict):
    api_key = os.getenv("NEBIUS_API_KEY")

    if not api_key:
        return generate_fallback_explanation(text, prediction_result)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"),
        )

        label = prediction_result["label"]
        confidence = prediction_result["confidence"]

        prompt = f"""
You are explaining the output of a trained sentiment analysis model.

User text:
{text}

Model prediction:
{label}

Model confidence:
{confidence}

Write a concise, helpful explanation for a non-technical user. Mention that this is a model estimate and may be wrong for sarcasm, slang, or missing context.
"""

        response = client.chat.completions.create(
            model=os.getenv("NEBIUS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
            messages=[
                {"role": "system", "content": "You explain ML sentiment predictions clearly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content

    except Exception:
        return generate_fallback_explanation(text, prediction_result)


def analyze_text(text: str, model_path=MODEL_PATH):
    is_valid, error_message = validate_user_input(text)

    if not is_valid:
        return {
            "status": "error",
            "message": error_message,
        }

    model = load_model(model_path)
    prediction_result = predict_sentiment(model, text)
    explanation = generate_llm_explanation(text, prediction_result)

    return {
        "status": "success",
        "input_text": text,
        "prediction": prediction_result,
        "explanation": explanation,
    }


def main():
    st.title("Sentiment Analysis Assistant")

    st.write(
        "Enter a tweet, review, or short comment. The trained ML model will classify "
        "the sentiment, and the assistant will explain the result."
    )

    user_text = st.text_area(
        "Text to analyze",
        placeholder="Example: I love the new update, it works so much better now!",
    )

    if st.button("Analyze Sentiment"):
        result = analyze_text(user_text)

        if result["status"] == "error":
            st.warning(result["message"])
            return

        prediction = result["prediction"]

        st.subheader("Prediction")
        st.write(f"Sentiment: **{prediction['label'].title()}**")

        if prediction["confidence"] is not None:
            st.write(f"Confidence: **{prediction['confidence']:.1%}**")

        st.subheader("Explanation")
        st.write(result["explanation"])


if __name__ == "__main__":
    main() 