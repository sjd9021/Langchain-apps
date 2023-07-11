import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredFileLoader
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")

openai.api_type = "azure"
openai.api_version = OPENAI_DEPLOYMENT_VERSION
openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_key = OPENAI_API_KEY

def get_text_chunks(pdf):
    loader = UnstructuredFileLoader(pdf)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=100, chunk_overlap=10, length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


pdf = "functionalsample.pdf"

# get pdf text

# get the text chunks
text_chunks = get_text_chunks(pdf) 

embeddings = OpenAIEmbeddings()
doc_search = Chroma.from_documents(text_chunks, embeddings)
chain = RetrievalQA.from_chain_type(llm=AzureOpenAI(model_kwargs={'engine':'samvit-chatbot2'}), chain_type='stuff', retriever = doc_search.as_retriever())
chain.run("hey how are you")