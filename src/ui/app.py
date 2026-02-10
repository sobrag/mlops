"""
Streamlit UI for News Credibility Checker.

Calls the Flask prediction API and displays results.
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:5001")


def call_predict_api(text: str, api_url: str) -> dict:
    """
    Call the prediction API and return the result.

    Returns a dict with either prediction keys
    (credibility_score, probability, label) or an "error" key.
    """
    if not text or not text.strip():
        return {"error": "Please enter a non-empty text."}

    try:
        response = requests.post(
            f"{api_url}/predict",
            json={"text": text},
            timeout=30,
        )
    except requests.ConnectionError:
        return {"error": "Connection error: could not reach the API."}
    except requests.Timeout:
        return {"error": "Request timed out. Please try again."}
    except requests.RequestException as exc:
        return {"error": f"Request failed: {exc}"}

    if response.status_code != 200:
        try:
            body = response.json()
            return {"error": body.get("error", f"Server error ({response.status_code})")}
        except Exception:
            return {"error": f"Server error ({response.status_code})"}

    return response.json()


def _score_color(score: float) -> str:
    """Return a hex color from red to green based on 0-100 score."""
    if score < 40:
        return "#e74c3c"
    if score < 70:
        return "#f39c12"
    return "#27ae60"


def main():
    st.set_page_config(page_title="News Credibility Checker", layout="centered")
    st.title("News Credibility Checker")

    text = st.text_area(
        "Paste the news article text below:",
        height=200,
        placeholder="Enter news article text here...",
    )

    if st.button("Analyze"):
        if not text or not text.strip():
            st.error("Please enter some text to analyze.")
            return

        with st.spinner("Analyzing..."):
            result = call_predict_api(text, API_URL)
            
        if "error" in result:
            st.error(result["error"])
            return

        score = result["credibility_score"]
        label = result["label"]
        probability = result["probability"]
        color = _score_color(score)

        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Credibility Score", f"{score:.1f} / 100")
        with col3:
            st.metric("Confidence", f"{probability * 100:.1f}%")

        st.progress(score / 100)
        st.markdown(
            f"<div style='text-align:center; font-size:1.2em; color:{color};'>"
            f"{'Likely Real' if label == 'real' else 'Likely Fake'}"
            f"</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
