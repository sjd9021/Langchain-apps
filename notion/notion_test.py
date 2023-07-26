"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

import openai
import os
from dotenv import load_dotenv
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




# Here we load in the data in the format that Notion exports it in.
# ps = list(Path("Notion_DB/").glob("**/*.md"))


# data = []
# sources = []
# for p in ps:
#     with open(p) as f:
#         data.append(f.read())
#     sources.append(p)

from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("Notion_Db")
# docs = loader.load()
print(loader.load_and_split())

# # Here we split the documents, as needed, into smaller chunks.
# # We do this due to the context limits of the LLMs.
# text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200,separator="\n")
# docs = []
# metadatas = []
# for i, d in enumerate(data):
#     splits = text_splitter.split_text(d)
#     docs.extend(splits)
#     metadatas.extend([{"source": sources[i]}] * len(splits))

# embedding=OpenAIEmbeddings(deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)
# # # Here we create a vector store from the documents and save it to disk.
# store = FAISS.from_texts(docs, embedding=embedding, metadatas=metadatas)

# store.save_local("notion/documentation/faiss_index")
# llm = AzureChatOpenAI(    
#                   deployment_name=OPENAI_DEPLOYMENT_NAME,
#                       model=OPENAI_MODEL_NAME,
#                       openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
#                       openai_api_version=OPENAI_DEPLOYMENT_VERSION,
#                       openai_api_key=OPENAI_API_KEY,
#                       temperature=0.0
#                      )
# chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=store.as_retriever())
# question = 'Tell me about Diversity in Blendle'
# result = chain({"question": question})
# print(f"Answer: {result['answer']}")
# print(f"Sources: {result['sources']}")