import pytesseract
from PIL import Image
import cv2
import re
import os
# Set the Tesseract command path
if os.name == "nt":  # For local development on Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:  # For deployment on Streamlit Cloud (Linux)
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6')
    return text

def clean_ingredients(text):
    """Clean the extracted text to get a list of ingredients after 'INGREDIENTS'."""
    ingredients_start = re.search(r'INGREDIENTS', text)
    if ingredients_start:
        text_after_ingredients = text[ingredients_start.end():]
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text_after_ingredients)
        ingredients = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        return ingredients
    else:
        return []

image_path = 'try3.png'  
extracted_text = extract_text_from_image(image_path)
ingredients = clean_ingredients(extracted_text)

# Print the list of ingredients
print(ingredients)
