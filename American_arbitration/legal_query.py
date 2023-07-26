import openai
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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

pdfs = os.scandir(r'C:\Users\SJ98023\OpenLang\American_arbitration\AAA')
loaders = []
for i in pdfs:
    loaders.append(PyPDFLoader(i.path))

docs = []
for loader in loaders:
    docs.extend(loader.load())
    
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
splits = text_splitter.split_documents(docs)

embedding=OpenAIEmbeddings(deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)
vectordb = FAISS.load_local("American_arbitration/American_arbitration/data/documentation/faiss_index", embeddings=embedding)
llm = AzureChatOpenAI(    
                  deployment_name=OPENAI_DEPLOYMENT_NAME,
                      model=OPENAI_MODEL_NAME,
                      openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                      openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                      openai_api_key=OPENAI_API_KEY,
                      temperature=0.0
                     )

template = """You are Legal Assistant bot, Your job is to answer any legal queries the user asks. Use the given context to answer the question below. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as detailed as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
---------
Question: {question}
Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
qa = RetrievalQA.from_chain_type(
llm=llm,
chain_type="refine",
retriever=vectordb.as_retriever(),
verbose=True
# prompt=QA_CHAIN_PROMPT
# chain_type_kwargs={
#     "prompt": QA_CHAIN_PROMPT,
#     # "memory": ConversationBufferMemory(
#     #     memory_key="history",
#     #     input_key="question")
# }
)
while True:
    question = str(input("you: "))
    result = qa({"query": question})
    print(result['result'])
