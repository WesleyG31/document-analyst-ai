# Streamlit dependencies
import os
import streamlit as st
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path, exceptions

# RAG dependencies
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Custom dependencies
from rag.rag import RAG_MODEL
from src.logger import get_logger
from src.custom_exception import CustomException

logger=get_logger(__name__)

vector_db_path=Path("tmp")/("vector_store")
vector_db_path.mkdir(parents=True, exist_ok=True)


def show_pdf_streamlit(pdf_path,file_name):
    try:
        logger.info("Get images from PDF to show in Streamlit")
        images_folder= Path(vector_db_path)/file_name/"images"
        os.makedirs(images_folder, exist_ok=True)

        image_paths=list(images_folder.glob("*.png"))
        if image_paths:
            try:
                logger.info("Images already exist")
                for img_path in image_paths:
                    image=Image.open(img_path)
                    st.sidebar.image(image, caption=f"Page {image_paths.index(img_path) + 1}", use_container_width=True)
                    logger.info("Loading images from PDF already existed successfully")
            except Exception as e:      
                logger.error(f"Error while opening image already existed: {e}")
                raise CustomException("Error while opening image already existed:",e)            
        else:
            try:
                logger.info("Images do not exist, converting PDF to images")               
                images = convert_from_path(pdf_path) 
                for i, image in enumerate(images):
                    img_path = images_folder / f"page_{i + 1}.png"
                    image.save(img_path, "PNG")  
                    st.sidebar.image(image, caption=f"Page {i + 1}", use_container_width=True)
                logger.info("converting PDF to images Successfully")  
            except Exception as e:      
                logger.error(f"Error while converting PDF to images: {e}")
                raise CustomException("Error while converting PDF to images:",e)            
    except Exception as e:
        logger.error(f"Error while showing PDF in Streamlit: {e}")
        raise CustomException("Error while showing PDF in Streamlit",e)

########################## Streamlit APP ##########################   
# Streamlit Title and layout
st.title("AI Document Analyst")

# upload documents
vector_db_options = [f.stem for f in Path(vector_db_path).iterdir() if f.is_dir()]
vector_db_options.append("Upload New Document") 
selected_vector_db = st.selectbox("Select Document", vector_db_options, index=0)

if selected_vector_db == "Upload New Document":
    uploaded_file= st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        st.sidebar.subheader(uploaded_file.name)

        temp_path= vector_db_path/f"temp_{uploaded_file.name}"
        document_binary= uploaded_file.read()
        with open(temp_path, "wb") as f:
            f.write(document_binary)

        show_pdf_streamlit(temp_path, uploaded_file.name.split('.')[0])  

        # Process the PDF
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                
                try:
                    logger.info("Processing PDF")

                    # saving the pdf
                    base_name = uploaded_file.name.split('.')[0]
                    pdf_path = vector_db_path / base_name / f"{base_name}.pdf"
                    with open(pdf_path,"wb") as f:
                        f.write(document_binary)
                    logger.info(f"PDF saved successfully at {pdf_path}")


                    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en",model_kwargs={"device": "cpu"})
                    rag_model= RAG_MODEL(pdf_path,base_name,embedding_model,vector_db_path)
                    vector_store=rag_model.run_vector()

                    #st.session_state.vector_store = vector_store
                    #st.session_state.rag_model = rag_model

                    st.success("PDF processed successfully")
                    logger.info("PDF processed successfully")

                    
                    Path(temp_path).unlink()
                except Exception as e:
                    logger.error(f"Error while processing PDF in streamlit: {e}")
                    raise CustomException("Error while processing PDF in streamlit",e)
                
# If the pdf was already processed dont process it again
elif selected_vector_db != "Upload New Document":

    try:
        logger.info("Loading PDF and vector store previously processed")

        pre_vector_db_path= Path(vector_db_path)/(selected_vector_db)/("db")
        
        if pre_vector_db_path.exists():
            try:
                logger.info(f"Check if the directory {pre_vector_db_path} exists")
                embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en",model_kwargs={"device": "cpu"})
                logger.info("embedding model created successfully")
                
                vector_store = Chroma(
                    persist_directory=str(pre_vector_db_path),
                    embedding_function=embedding_model
                )

                logger.info("Directory exists and vector store loaded successfully")
            except Exception as e:
                    logger.error(f"Error while checking if the directory exists {pre_vector_db_path}: {e}")
                    raise CustomException("Error while checking if the directory exists",e)

            pdf_path = vector_db_path / selected_vector_db / f"{selected_vector_db}.pdf"
            if pdf_path.exists():
                show_pdf_streamlit(pdf_path,selected_vector_db)

            rag_model= RAG_MODEL(pdf_path,selected_vector_db,embedding_model,vector_db_path)
        
        logger.info("PDF and vector store loaded successfully")
    except Exception as e:
        logger.error(f"Error while loading PDF and vector store: {e}")
        raise CustomException("Error while loading PDF and vector store",e)
    
# Define the assistant
assistant= st.text_input("What kind of assistance do you need?", placeholder="e.g. finance")

# Define the LLM 
LLM_options= ["deepseek/deepseek-chat-v3-0324:free","meta-llama/llama-3.3-8b-instruct:free","qwen/qwen3-0.6b-04-28:free","meta-llama/llama-4-maverick:free"]
LLM_used = st.selectbox("Which LLM do you want to use?", LLM_options)

# Define the question
question = st.text_input("Enter your question:", placeholder="e.g., What is the company's revenue?")

# RAG

if st.button("Submit Question") and question and selected_vector_db != "Upload New Document":
    with st.spinner("Answering your question..."):
        try:
            logger.info("Answering question in streamlit")     

            if not vector_store or not rag_model:
                st.error("Please process or load a document first.")
            else:
                retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})
                
                chain = rag_model.rag_chain(retriever, assistant,LLM_used)
                response_placeholder = st.empty()
            
                response = ""
                for chunk in chain.stream(question):
                    response += chunk  
                    response_placeholder.markdown(response.replace('$', '\\$'))
        except Exception as e:
            logger.error(f"Error while answering question in streamlit: {e}")
            raise CustomException("Error while answering question in streamlit",e)
        
