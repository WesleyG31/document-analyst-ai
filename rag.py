import os
from pathlib import Path
from dotenv import load_dotenv

# Vector DB dependencies
import chromadb

# Pytorch 
import torch 

# Langchain dependencies 
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import MarkdownHeaderTextSplitter
from docling.document_converter import DocumentConverter
from langchain_openai import ChatOpenAI

# Custom dependencies
from src.logger import get_logger
from src.custom_exception import CustomException



load_dotenv()
logger = get_logger(__name__)

class RAG_MODEL:
    def __init__(self, file_path,filename,embedding_model):
        self.file_path=file_path
        self.filename=filename
        self.embedding_model=embedding_model
        self.vector_db_path="/chroma_db"
        self.OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")

    def load_convert_document(self):
        try:
            logger.info(f"Loading and converting document: {self.file_path}")
            converter=DocumentConverter()
            result= converter.convert(self.file_path)
            return result.document.export_to_markdown()
        except Exception as e:
            logger.error(f"Error while loading and converting document: {e}")
            raise CustomException("Error while loading and converting document",e)

    def split_document(self,markdown_content):
        try:
            logger.info("Splitting document")
            headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
            return splitter.split_text(markdown_content)
        except Exception as e:
            logger.error(f"Error while splitting document: {e}")
            raise CustomException("Error while splitting document", e)

    def vector_store(self,chunks):
        try:
            logger.info(f"Creating Vector Store: {self.filename}")
            persist_directory= Path(self.vector_db_path)/self.filename

            if persist_directory.exists():
                vector_store= Chroma(
                    persist_directory=str(persist_directory),
                    embedding_function=self.embedding_model
                )
            else:
                vector_store= Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embedding_model,
                    persist_directory=str(persist_directory)
                )
                vector_store.persist()
            return vector_store
        except Exception as e:
            logger.error(f"Error while creating vector store: {e}")
            raise CustomException("Error while creating vector store",e)

    def rag_chain(self,retriever,assistant):
        try:
            logger.info("Creating RAG chain")
            prompt = """ You are an assistant for {assistant}. Use the retrieved context to answer questions. 
            If you don't know the answer, say so.
            Always answer in a professional tone. 
            Question: {question}
            Context: {context}
            Answer: """
            prompt_template = ChatPromptTemplate.from_template(prompt)
            
            llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                openai_api_key=self.OPENROUTER_API_KEY,
                model="deepseek/deepseek-r1:free",  
            )
            return (
                {"assistant": assistant,
                "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), 
                "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
        except Exception as e:
            logger.error(f"Error using RAG Chain: {e}")
            raise CustomException("Error using RAG Chain",e)
    
    def run_vector(self):
        try:
            logger.info("Creating Vector Store Pipeline")
            markdown_content = self.load_convert_document()
            chunks = self.split_document(markdown_content)
            vector_store = self.vector_store(chunks)
            logger.info("Vector Store created successfully")
            return vector_store
        except Exception as e:
            logger.error(f"Error in Vector Store Pipeline: {e}")
            raise CustomException("Error in Vector Store Pipeline",e)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en",model_kwargs={"device": device})
    file_path=r"C:\All files\document-analyst-ai\module-0.pdf"
    filename="module-0"
    assistant="Cloud Expert"
    question="What are the Course prerequisites?"
    rag_model= RAG_MODEL(file_path,filename,embedding_model)
    vector_store=rag_model.run_vector()
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5})
    chain = rag_model.rag_chain(retriever, assistant)
    response = chain.invoke(question)
    print(response)
        