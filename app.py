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
st.set_page_config(page_title="Chatbot Hadis", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Hadith Question Answering App💬")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Klik button to processing file"}
    ]

with st.sidebar:
    st.title('🦙💬 Hadit Chatbot')

if st.button("Process"):
    st.text("Processing ...")    
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
    if 'memory' not in st.session_state:

         st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

    if 'prompt' not in st.session_state:

         st.session_state.prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

    response = f"Processing done.!"
    st.session_state.messages = [{"role": "assistant", "content": response}]
    
if user_question := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
def get_text():
    global input_text
    input_text = st.text_input("Ask your Question", key="input")
    return input_text
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Pilihlah Kitab Apa yang Anda ingin tanyakan"}
    ]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0, streaming=True)
            chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        chain_type="stuff",
                        retriever= st.session_state.compression_retriever,
                        verbose=True,
                        max_tokens_limit = 3000,
                        combine_docs_chain_kwargs={"prompt": st.session_state.prompt},
                        memory= st.session_state.memory,
                        return_source_documents=True,
                    )
            # st_callback = StreamlitCallbackHandler(st.container())
            res = chain(user_question)#, callbacks=[st_callback])
            answer = res["answer"]
            
            placeholder = st.empty()
            placeholder.markdown(answer)
            message = {"role": "assistant", "content": answer}
            st.session_state.messages.append(message) # Add response to message history 
