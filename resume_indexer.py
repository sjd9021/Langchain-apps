from langchain.llms import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
import os
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain,RetrievalQA,ConversationChain
import PyPDF2
import io
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")

#init Azure OpenAI
openai.api_type = "azure"
openai.api_version = OPENAI_DEPLOYMENT_VERSION
openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_key = OPENAI_API_KEY
llm = AzureChatOpenAI(    
                    deployment_name=OPENAI_DEPLOYMENT_NAME,
                        model=OPENAI_MODEL_NAME,
                        openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                        openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                        openai_api_key=OPENAI_API_KEY,
                        temperature=0.0
                        )

def main():
        
    with open('data.txt', 'r', encoding='utf-8') as file:
        data = file.read()
        data = str(data)
    text_splitter = CharacterTextSplitter(
        separator=" ", chunk_size=1000, chunk_overlap=500, length_function=len
    )
    splits = text_splitter.split_text(data)
    embedding=OpenAIEmbeddings(deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)
    vectordb = FAISS.from_texts(
    texts=splits,
    embedding=embedding
)
    vectordb.save_local("database/documentation/faiss_index")

main()