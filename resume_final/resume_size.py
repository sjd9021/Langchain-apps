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
import pickle
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
def extract_text_from_binary(file):
    reader = PyPDF2.PdfReader(file)
    num_pages = len(reader.pages)
    text = ""

    for page in range(num_pages):
        current_page = reader.pages[page]
        text += current_page.extract_text()
    return text
def get_templates(link):
    resume = extract_text_from_binary(link)

    template = """Format the provided resume to this YAML template:
            ---
        name: ''
        phoneNumbers:
        - ''
        websites:
        - ''
        emails(@):
        - ''
        summary: ''
        education:
        - University: ''
        degree: ''
        fieldOfStudy: ''
        startDate: ''
        endDate: ''
        workExperience:
        - company: ''
        position: ''
        startDate: ''
        endDate: ''
        skills:
        - primary skills: ''
        secondary skills: ''
        certifications:
        - name: ''
        {chat_history}
        {human_input}"""


    prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"],
            template=template
        )

    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory
        )

    res = llm_chain.predict(human_input=resume)
    return res

# print(get_templates("resume/Ankit_Kamanalli(75132).pdf"))
def main():
    master_list=[]
    resume = os.scandir("resume")
    for x in resume:
        answer = get_templates(x.path)
        master_list.append(answer)
        print(answer)
    with open("test", "wb") as fp:   #Pickling
      pickle.dump(master_list, fp)

    
def test():
    file = open('testing_data.txt','w', encoding='utf-8')
    with open("test", "rb") as fp:   # Unpickling
        b = pickle.load(fp)
    
    # print(len(b[2].split("\n")))
    max = 0
    for i in b:
        c = 0
        for x in i.split('\n'):
            c = c + 1
        if c > max:
            max = c

    for resumes in b:
        counter = 0
        for lines in resumes.split('\n'):
            counter = counter + 1
        while(counter < max):
            resumes = resumes + "-----------------------------------------------------------------------\n"
            counter += 1
        file.write(resumes+"\n")

    file.close()
    
test()