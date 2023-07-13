from langchain.llms import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
import os
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
# def extract_text_from_binary(file):
#     reader = PyPDF2.PdfReader(file)
#     num_pages = len(reader.pages)
#     text = ""

#     for page in range(num_pages):
#         current_page = reader.pages[page]
#         text += current_page.extract_text()
#     return text
# def get_templates(link):
#     resume = extract_text_from_binary(link)

#     template = """Format the provided resume to this YAML template:
#             ---
#         name: ''
#         phoneNumbers:
#         - ''
#         websites:
#         - ''
#         emails(@):
#         - ''
#         summary: ''
#         education:
#         - University: ''
#         degree: ''
#         fieldOfStudy: ''
#         startDate: ''
#         endDate: ''
#         workExperience:
#         - company: ''
#         position: ''
#         startDate: ''
#         endDate: ''
#         skills:
#         - primary skills: ''
#         secondary skills: ''
#         certifications:
#         - name: ''
#         {chat_history}
#         {human_input}"""


#     prompt = PromptTemplate(
#             input_variables=["chat_history", "human_input"],
#             template=template
#         )

#     memory = ConversationBufferMemory(memory_key="chat_history")

#     llm_chain = LLMChain(
#             llm=llm,
#             prompt=prompt,
#             memory=memory,
#         )

#     res = llm_chain.predict(human_input=resume)
#     return res

def main():
  
    embeddings=OpenAIEmbeddings(deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)
    vectordb =  FAISS.load_local("database/documentation/faiss_index", embeddings)
    template = """You are Talent Acquisition bot, Your job is to find out details about candidates from their resumes and suggest Suitable Candidates. Use the given context to answer the question below. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    ---------
    {history}
    ---------
    Question: {question}
    Helpful Answer:"""
    # QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["history", "context", "question"],template=template)


#     memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True
# )
    
#     # retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#     qa = ConversationalRetrievalChain.from_llm(
#     llm,
#     retriever=retriever,
#     memory=memory,
#     chain_type="stuff",
#     combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
# )
    
    
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=vectordb.as_retriever(),
    #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    # )
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vectordb.as_retriever(),
    chain_type_kwargs={
        "prompt": QA_CHAIN_PROMPT,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    }
)
    while True:
        question = str(input("you: "))
        result = qa({"query": question})
        print(result['result'])

main()