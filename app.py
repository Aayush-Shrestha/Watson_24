import streamlit as st
import pandas as pd
import tempfile
import os
import cv2
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from ocr import extract_text_from_image

# Load Hugging Face Token
token = os.getenv("HF_TOKEN")

# Load Model and Tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
try:
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

# Load CSV files for interactions and allergies
csv_file = "interactions.csv"
try:
    data_intr = pd.read_csv(csv_file)
    unique_drug_names = data_intr["Drug_Name"].unique()
except FileNotFoundError:
    st.error("The 'interactions.csv' file is missing. Please upload it to proceed.")
    st.stop()

try:
    data_aller = pd.read_csv("allergies.csv")
    data_aller["food_normalized"] = data_aller["food"].str.lower()  # Normalize for matching
except FileNotFoundError:
    st.error("The 'allergies.csv' file is missing. Please upload it to proceed.")
    st.stop()

def match_allergy(row):
    food_words = set(row["food_normalized"].split())
    return not ocr_tokens.isdisjoint(food_words)

# Streamlit UI configuration
st.set_page_config(
    page_title="SafeConsume",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header("Drug-Interaction and Allergy Analysis")
st.markdown("---")

selected_drug_name = st.selectbox("Select a Drug Name:", unique_drug_names)
text_input2 = st.text_input("Enter your allergy (You are allergic to):", value="na")
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if st.button("Submit"):
    st.write("Selected Drug Name:", selected_drug_name)
    st.write("Allergy Input:", text_input2)
    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_image.getvalue())
            temp_image_path = tmp_file.name

        # Validate image reading
        image = cv2.imread(temp_image_path)
        if image is None:
            st.error("Uploaded image cannot be read. Please check the file and try again.")
            st.stop()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        extracted_text = extract_text_from_image(temp_image_path)
        os.remove(temp_image_path)

        st.write("Extracted Text from Image:")
        st.text(extracted_text)

        prompt = {
            "role": "system",
            "content": "You are a helpful assistant who separates ingredients from text. Just reply with the list and nothing else."
        }
        user_input = {
            "role": "user",
            "content": f"Separate the ingredients from this text:\n{extracted_text}"
        }
        messages = [prompt, user_input]

        try:
            model_input = tokenizer(messages, return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**model_input, max_length=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "INGREDIENTS:" in generated_text:
                ingredients_text = generated_text.split("INGREDIENTS:")[-1].strip()
                cleaned_ingredients = [ingredient.strip() for ingredient in ingredients_text.split(",")]
            else:
                st.warning("No ingredients found in the generated text.")
                cleaned_ingredients = []

            ocr_tokens = set(" ".join(cleaned_ingredients).lower().split())

            matched_allergens = data_aller[data_aller.apply(match_allergy, axis=1)]
            if not matched_allergens.empty:
                st.write("**Matching Allergens:**")
                for _, row in matched_allergens.iterrows():
                    aller = row["food"]
                    explain = row["allergy"]

                prompt = {
                    "role": "system",
                    "content": (
                        "You will be given what the patient is allergic to and what he is about to consume along with its effect. "
                        "Your task is to only reply in a sentence with either or not he will suffer an allergic reaction."
                    )
                }
                user_input = {
                    "role": "user",
                    "content": (
                        f"You have a patient who is allergic to {text_input2}. "
                        f"He is about to eat something containing a notable allergen ingredient: {aller}, with the following Allergy Info: {explain}."
                    )
                }
                messages = [prompt, user_input]

                model_input = tokenizer(messages, return_tensors="pt", padding=True, truncation=True)
                outputs = model.generate(**model_input, max_length=30, do_sample=True, temperature=0.7, top_k=1, top_p=1.0)
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                st.markdown(
                    f"""
                    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
                        <p style="font-size: 16px; margin: 0;">{generated_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("*Reason*", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style="border: 2px solid #ff6ec7; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
                        <p style="font-size: 16px; margin: 0;">{explain}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("No allergens found in the OCR ingredient list.")

        except Exception as e:
            st.error(f"Error in generating text or processing input: {e}")

    else:
        st.warning("Please upload an image!")

st.write("---")
