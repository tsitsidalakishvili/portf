import streamlit as st
from PIL import Image
import numpy as np
import cv2
from pdf2image import convert_from_bytes
from rapidocr_onnxruntime import RapidOCR

# Initialize RapidOCR
ocr_engine = RapidOCR()

# Helper function to convert PIL image to OpenCV format and ensure compatibility
def pil_to_cv2_compatible(pil_image):
    # Convert to RGB if in RGBA format to remove alpha channel
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR for OpenCV
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

# Perform OCR with error handling and preprocessing
def perform_ocr(image):
    try:
        # Ensure the image is compatible with OpenCV and the OCR engine
        open_cv_image = pil_to_cv2_compatible(image)
        result, _ = ocr_engine(open_cv_image)
        if result is None or not result:
            st.error("OCR processing returned no results. This could be due to an unrecognizable image format or OCR engine limitations.")
            return ""
        extracted_text = ' '.join([res[1] for res in result])
    except Exception as e:
        st.error(f"An error occurred during OCR processing: {e}")
        extracted_text = ""
    return extracted_text

# Helper function to convert PDF pages to images with proper preprocessing
def pdf_to_images(pdf_file):
    images = convert_from_bytes(pdf_file.read())
    # Ensure images are in a compatible format for OCR
    images = [image.convert('RGB') for image in images]
    return images

# Streamlit UI for uploading and processing images or PDFs
st.title("OCR")
uploaded_file = st.file_uploader("Upload an image or PDF file", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file iscd not None:
    if uploaded_file.type == "application/pdf":
        with st.spinner('Processing PDF...'):
            images = pdf_to_images(uploaded_file)
            for i, image in enumerate(images):
                st.image(image, caption=f'Page {i+1}', use_column_width=True)
                extracted_text = perform_ocr(image)
                st.text_area(f"Extracted Text (Page {i+1})", extracted_text, height=200)
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        extracted_text = perform_ocr(image)
        st.text_area("Extracted Text", extracted_text, height=200)
