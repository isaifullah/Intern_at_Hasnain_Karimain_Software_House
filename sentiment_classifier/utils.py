import streamlit as st
import joblib
import os

# 1. Page Configuration
st.set_page_config(page_title="Sentiment Analysis AI", page_icon="🧠")

# 2. Simple Load Function
@st.cache_resource
def load_assets():
    # We are using direct filenames. 
    # Make sure these match your file names in the folder EXACTLY.
    model = joblib.load('sentiment_classifier_model.pkl')
    vectorizer = joblib.load('best_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, vectorizer, label_encoder

# 3. Load the files
try:
    classifier, tfidf, encoder = load_assets()
except FileNotFoundError as e:
    st.error(f"❌ File Not Found: {e.filename}")
    st.info("Check if your files are named correctly (e.g., check for 'classifer' vs 'classifier')")
    st.stop()
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# 4. User Interface
st.title("🧠 Sentiment Analysis System")
st.markdown("---")

user_input = st.text_area("Enter text to analyze:", placeholder="The experience was good but the wait was long...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Step 1: Vectorize
        transformed_input = tfidf.transform([user_input])
        
        # Step 2: Predict
        numeric_prediction = classifier.predict(transformed_input)
        
        # Step 3: Decode to Human Word
        final_label = encoder.inverse_transform(numeric_prediction)[0]
        
        # 5. Show Results
        st.subheader("Result:")
        if final_label == "Positive":
            st.success(f"### {final_label} ✨")
            st.balloons()
        elif final_label == "Negative":
            st.error(f"### {final_label} ❌")
        elif final_label == "Mixed":
            st.warning(f"### {final_label} ⚖️")
        else:
            st.info(f"### {final_label} 😐")
            
        # Recommendation Logic
        st.markdown("---")
        st.write("**Recommendation:**")
        if final_label == "Positive":
            st.write("Keep it up! This is exactly what users want to see.")
        elif final_label == "Negative":
            st.write("Consider investigating the issues mentioned to improve user satisfaction.")
        else:
            st.write("Try to gather more specific feedback to identify areas for improvement.")
            
    else:
        st.warning("Please enter some text first!")