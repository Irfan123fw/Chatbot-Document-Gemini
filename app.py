import streamlit as st
from langchain.vectorstores import Chroma
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

DATA_PATH = 'content/'

def initialize_system():
    if 'responses' not in st.session_state:
         st.session_state['responses'] = ["Processing"]
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    docsearch = Chroma.from_documents(texts,embeddings)


    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

    response = f"Processing done. Silahkan Ajukan Pertanyaan Anda !"
    st.session_state['responses'] = [(response)]
    return docsearch, memory, prompt

st.title("Question Answering Pdf Using Gemini-pro")

if "initialized" not in st.session_state:
    st.session_state.docsearch, st.session_state.memory, st.session_state.prompt = initialize_system()
    st.session_state.initialized = True
if 'requests' not in st.session_state:
    st.session_state['requests'] = []
# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()
# Initialize variables and objects
# Streamlit main loop
with textcontainer:
    query = st.text_input("Ask your question:")
    if st.button("Ask"):
        docsearch = st.session_state.docsearch
        memory = st.session_state.memory
        prompt = st.session_state.prompt

        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, max_output_tokens=2048)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            verbose=True,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        results =  chain.run(query)
        print(results)
        st.session_state.requests.append(query)
        st.session_state.responses.append(results) 
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
