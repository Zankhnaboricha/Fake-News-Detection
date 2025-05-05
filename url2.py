import streamlit as st
import joblib
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from serpapi import GoogleSearch
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Load the model and vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# --- Helper Functions ---

def extract_with_newspaper(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None

def extract_with_bs4(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = "\n".join(p.get_text() for p in paragraphs if len(p.get_text()) > 50)
            return text
        return None
    except:
        return None

def predict_news(text):
    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)
    proba = model.predict_proba(transformed)
    return prediction[0], proba[0]

def fact_check_with_serpapi(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 5
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        snippets = []

        if "organic_results" in results:
            for result in results["organic_results"]:
                if "snippet" in result:
                    snippets.append(result["snippet"])
        return snippets
    except Exception as e:
        return []

# --- Streamlit UI ---

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.write("Analyze a news article by either pasting the **text** or providing a **URL**.")

tab1, tab2 = st.tabs(["📝 Text Input", "🌐 URL Input"])

# --- TEXT INPUT ---
with tab1:
    input_text = st.text_area("Enter the full news article or claim:")

    if st.button("Check Text"):
        if input_text.strip():
            if len(input_text.split()) < 10:
                # Short query – use search-based fact-check
                st.info("🔍 Using real-time fact-check from Google...")
                snippets = fact_check_with_serpapi(input_text)

                if snippets:
                    match_found = any(all(word in s.lower() for word in input_text.lower().split()) for s in snippets)
                    st.markdown("#### 🔎 Search Snippets:")
                    for s in snippets:
                        st.markdown(f"> {s}")

                    if match_found:
                        st.success("✅ The claim appears to be supported by trusted sources.")
                    else:
                        st.warning("⚠️ Could not verify the claim in trusted sources.")
                else:
                    st.error("🚫 Could not fetch search results.")
            else:
                # Long article – use ML model
                label, proba = predict_news(input_text)
                st.info(f"🧠 Confidence (Real): {proba[1]:.2f} | (Fake): {proba[0]:.2f}")
                if label == 1:
                    st.success("✅ The News is Real!")
                else:
                    st.error("❌ The News is Fake!")
        else:
            st.warning("⚠️ Please enter some text.")

# --- URL INPUT ---
with tab2:
    url = st.text_input("Paste a news article URL:")

    if st.button("Check URL"):
        if url.strip():
            st.info("🔄 Extracting article...")
            article_text = extract_with_newspaper(url)
            if not article_text:
                article_text = extract_with_bs4(url)

            if article_text:
                st.success("✅ Article extracted successfully!")
                st.write(article_text[:500] + "..." if len(article_text) > 500 else article_text)

                label, proba = predict_news(article_text)
                st.info(f"🧠 Confidence (Real): {proba[1]:.2f} | (Fake): {proba[0]:.2f}")
                if label == 1:
                    st.success("✅ The News is Real!")
                else:
                    st.error("❌ The News is Fake!")
            else:
                st.error("🚫 Could not extract content from the URL. Try a different source like BBC or Reuters.")
        else:
            st.warning("⚠️ Please enter a URL.")
