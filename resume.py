
import openai
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,RetrievalQA


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


pdfs = os.scandir('resume')
loaders = []
for i in pdfs:
    loaders.append(PyPDFLoader(i.path))

docs = []
for loader in loaders:
    docs.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 600,
    chunk_overlap = 100
)
splits = text_splitter.split_documents(docs)
# len(splits)
# 33
embedding=OpenAIEmbeddings(deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)
vectordb = FAISS.from_documents(
    documents=splits,
    embedding=embedding
)

vectordb = FAISS.load_local("dbs/documentation/faiss_index", embedding)
llm = AzureChatOpenAI(    
                  deployment_name=OPENAI_DEPLOYMENT_NAME,
                      model=OPENAI_MODEL_NAME,
                      openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                      openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                      openai_api_key=OPENAI_API_KEY,
                      temperature=0.0
                     )
# Build prompt
template = """You are Talent Acquisition bot, Your job is to find out details about candidates from their resumes and suggest Suitable Candidates. Use the given context to answer the question below. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 1})
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    chain_type="stuff",
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# qa= RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectordb.as_retriever(),
#     chain_type="map_reduce"
# )


# def ask_question(question):
#     # result =qa({"question": question})
#     # # print("Question:", question)
#     # print("Answer:", result)
#     result = qa.run(question)
#     print(result)
    
    
# question = "Does Ankit work at Zensar"
# result = qa.run(question)
# print(result)   
while True:
        question = str(input("you: "))
        result = qa.run(question)
        print(result)
