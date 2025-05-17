import os
import streamlit as st
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from rag.rag import RAG_MODEL
from src.logger import get_logger
from src.custom_exception import CustomException

logger=get_logger(__name__)

vector_db_path=r"C:/All files/document-analyst-ai/vector_store"

def show_pdf_streamlit(pdf_path,file_name):
    try:
        logger.info("Get images from PDF to show in Streamlit")
        images_folder= Path(vector_db_path)/file_name/"images"
        os.makedirs(images_folder, exist_ok=True)

        image_paths=list(images_folder.glob("*.png"))
        if image_paths:
            for img_path in image_paths:
                image=Image.open(img_path)
                st.sidebar.image(image, caption=f"Page {image_paths.index(img_path) + 1}", use_container_width=True)
        else:
            images = convert_from_path(pdf_path)  # This will render all pages by default
            for i, image in enumerate(images):
                img_path = images_folder / f"page_{i + 1}.png"
                image.save(img_path, "PNG")  # Save image to disk
                st.sidebar.image(image, caption=f"Page {i + 1}", use_container_width=True)
    except Exception as e:
        logger.error(f"Error while showing PDF in Streamlit: {e}")
        raise CustomException("Error while showing PDF in Streamlit",e)
    
########################## Streamlit APP ##########################   
# Streamlit Title and layout
st.title("AI Document Analyst for what you need")

# upload documents
vector_db_options = [f.stem for f in Path(vector_db_path)]
vector_db_options.append("Upload New Document") 
selected_vector_db = st.selectbox("Select Document", vector_db_options, index=0)

if selected_vector_db == "Upload New Document":
    uploaded_file= st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        st.sidebar.subheader