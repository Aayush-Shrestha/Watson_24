import streamlit as st
import pandas as pd
import tempfile
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from ocr import extract_text_from_image

token = os.getenv("HF_TOKEN")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Load CSV files for interactions and allergies
csv_file = "interactions.csv"
data_intr = pd.read_csv(csv_file)
unique_drug_names = data_intr["Drug_Name"].unique()

# Handle allergies CSV file with exception handling
try:
    data_aller = pd.read_csv("allergies.csv")  # Assuming you have this CSV file with a 'food' column
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
    layout="wide",  # Set layout to 'wide'
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
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    st.write("-" * 50)

    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_image.getvalue())
            temp_image_path = tmp_file.name

        extracted_text = extract_text_from_image(temp_image_path)
        
        prompt = {
            "role": "system",
            "content": "You are a helpful assistant who separates ingredients from text. Just reply with the list and nothing else"
        }
        user_input = {
            "role": "user",
            "content": f"Separate the ingredients from this text:\n{extracted_text}"
        }
        messages = [prompt, user_input]
        model_input = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(model_input, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        generated_text = outputs[0]["generated_text"]

        if "INGREDIENTS:" in generated_text:
            ingredients_text = generated_text.split("INGREDIENTS:")[-1].strip()
            cleaned_ingredients = [ingredient.strip() for ingredient in ingredients_text.split(",")]
        else:
            st.warning("No ingredients found in the generated text.")

        ocr_tokens = set(" ".join(cleaned_ingredients).lower().split())
        data_aller['food_normalized'] = data_aller['food'].str.lower()

        matched_allergens = data_aller[data_aller.apply(match_allergy, axis=1)]
        if not matched_allergens.empty:
            st.write("**Matching Allergens:**")
            for _, row in matched_allergens.iterrows():
                aller = row['food']
                explain = row['allergy']
            
            prompt = {
                "role": "system",
                "content": (
                    "You will be given what the patient is allergic to and what he is about to consume along with its effect. Your task is to only reply in a sentence with either or not he will suffer allergic reaction"
                )
            }
            user_input = {
                "role": "user",
                "content": (
                    f"You have a patient who is allergic to {text_input2}. "
                    f"He is about to eat something containing a notable allergen ingredient: {aller}, with the following Allergy Info: {explain}. generate the reply with no more than 20 words "
                )
            }
            messages = [prompt, user_input]
            model_input = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = pipe(model_input, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=1, top_p=1.0)
            generated_text = outputs[0]["generated_text"]
            generated_text = generated_text.split("<|assistant|>")[-1].strip()
            generated_text = generated_text.split('.')[0].strip()

            st.markdown(
                f"""
                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
                    <p style="font-size: 16px; margin: 0;">{generated_text}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Reason section
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
        st.write("-" * 50)
        df = data_intr[data_intr["Drug_Name"] == selected_drug_name]

        df['food_normalized'] = df['Food_Herb_Name'].str.lower()
        matched_intr = df[df.apply(match_allergy, axis=1)]
        if not matched_intr.empty:
            st.write("**Drug-Food Interaction:**")
            for _, row in matched_intr.iterrows():
                ingr = row['Food_Herb_Name']
                eff = row['Effect']
                conc = row['Conclusion']

            prompt = {
                "role": "system",
                "content": (
                    "Your task is to summarize the Food drug interaction that the patient may experience. Keep it very concise and do not add things on your own. Maximum 30 words."
                )
            }
            user_input = {
                "role": "user",
                "content": (
                    f"You have a patient who takes {selected_drug_name} , he is about to eat something containing {ingr}, which may affect the medication ({selected_drug_name}) as discussed: {conc}. "
                    f"Summarize only for the Food Drug interaction that may occur while excluding any additional details. Keep it very concise and do not add things on your own"
                )
            }
            messages = [prompt, user_input]
            model_input = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = pipe(model_input, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            generated_text = outputs[0]["generated_text"]
            generated_text = generated_text.split("<|assistant|>")[-1].strip()

            st.write(generated_text)
        else:
            st.warning("No allergens found in the OCR ingredient list.")
        st.write("-" * 50)

        os.remove(temp_image_path)
    else:
        st.warning("Please upload an image!")
