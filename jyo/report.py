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
from langchain.chains.question_answering import load_qa_chain
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


pdfs = ['annual_report.pdf']

annual_reports = []
for pdf in pdfs:
    loader = PyPDFLoader(pdf)
    # Load the PDF document
    document = loader.load_and_split()        
    # Add the loaded document to our list
    annual_reports.append(document)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(annual_reports[0])


# chunked_annual_reports = []
# for annual_report in annual_reports:
#     # Chunk the annual_report

#     # Add the chunks to chunked_annual_reports, which is a list of lists
#     chunked_annual_reports.append(texts)
#     print(f"chunked_annual_report length: {len(texts)}")

embedding=OpenAIEmbeddings(deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)
# db = FAISS.from_documents(documents=texts, embedding=embedding)
# db.save_local("jyo/documentation/faiss_index")
db = FAISS.load_local("jyo/documentation/faiss_index", embedding)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever())
x = qa.run("what is Mankind Pharmas total debt?")
print(x)



from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir="docs/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()