import streamlit as st
import pickle
import re
from nltk.stem import WordNetLemmatizer
import nltk
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloder.DownloadError:
    nltk.download('wordnet')
# You must make sure NLTK is installed in your Streamlit environment: pip install streamlit nltk scikit-learn

# --- 1. Load the Model and Vectorizer ---
@st.cache_resource
def load_assets():
    """Loads the trained model and the TF-IDF vectorizer."""
    try:
        # Load Model
        with open('spam_classifier_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        # Load Vectorizer
        with open('tfidf_vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
        
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model or Vectorizer file not found. Make sure 'spam_classifier_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
        return None, None

model, vectorizer = load_assets()

# --- 2. Text Preprocessing Function ---
# This function MUST exactly match the preprocessing steps used during training!
def clean_and_lemmatize(text):
    """Performs cleaning and lemmatization on input text."""
    if not hasattr(clean_and_lemmatize, 'lemmatizer'):
        clean_and_lemmatize.lemmatizer = WordNetLemmatizer()
    
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = [clean_and_lemmatize.lemmatizer.lemmatize(word) for word in text.split()]
    
    return " ".join(words)

# --- 3. Streamlit UI and Prediction Logic ---
def main():
    st.title("✉️ SMS/Email Spam Classifier")
    st.write("Enter a message below to classify it as 'Ham' (not spam) or 'Spam'.")

    # Text Area for User Input
    user_input = st.text_area("Enter your message:", height=150)
    
    if st.button("Classify Message"):
        if model is None or vectorizer is None:
            return

        if user_input:
            with st.spinner('Analyzing message...'):
                # 1. Preprocess the input text
                cleaned_input = clean_and_lemmatize(user_input)
                
                # 2. Vectorize the cleaned text (Must use the loaded vectorizer)
                # Note: We use .transform(), NOT .fit_transform()
                input_vectorized = vectorizer.transform([cleaned_input])
                
                # 3. Predict the category
                prediction = model.predict(input_vectorized)
                
                # 4. Display the result
                result = prediction[0]
                
                st.subheader("Prediction Result:")
                if result == 'spam':
                    st.error(f"⚠️ This message is classified as: *{result.upper()}*")
                    st.write("Be cautious! This content appears to be unsolicited or promotional.")
                elif result == 'ham':
                    st.success(f"✔️ This message is classified as: *{result.upper()}*")
                    st.write("This content appears to be legitimate.")
                else:
                    st.info(f"Classification: {result}")
main()