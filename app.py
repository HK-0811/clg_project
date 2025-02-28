import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os
load_dotenv()

hf_api_token = os.getenv("HUGGINGFACE_API_KEY")
def get_vectorstore(url):
    # Get the url text
    loader = WebBaseLoader(url)
    documents = loader.load()
    
    # Split document to chunks
    text_splitter = RecursiveCharacterTextSplitter()
    doc_chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    embedding_model = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(doc_chunks, embedding_model)
    
    return vector_store

def get_context_retriever_chain(vector_store):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceEndpoint(repo_id=model_id, max_length=128, temperature=0.4, huggingfacehub_api_token=hf_api_token)
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in to get information relevant to the conversation.")
    ])
    
    return create_history_aware_retriever(llm=llm, retriever=retriever, prompt=prompt)

def get_conversation_rag_chain(retrieved_chain):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceEndpoint(repo_id=model_id, max_length=128, temperature=0.4, huggingfacehub_api_token=hf_api_token)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Provide a brief, one-sentence answer to the user's question based on the context below. 
        Do not include any additional information or context in your response.
        Context: {context}"""),
        ("human", "{input}")
    ])
    
    stuff_document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retrieved_chain, stuff_document_chain)

def get_response(user_query):
    retrieved_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversation_rag_chain(retrieved_chain)
    
    response = conversation_rag_chain.invoke({
        "chat": st.session_state.chat_history,
        "input": user_query
    })
    
    return response.get('answer')

# App configuration
st.set_page_config(
    page_title="AI Website Chat Assistant",
    page_icon="💬",
    layout="wide"
)

st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
        }
        .stTextInput > div > div > input {
            background-color: #262730;
            color: white;
            border: 1px solid #4a4a4a;
        }
        .stTextInput > div > div > input:focus {
            border-color: #00acee;
            box-shadow: 0 0 0 2px rgba(0, 172, 238, 0.2);
        }
        .stButton > button {
            width: 100%;
            border-radius: 20px;
            background-color: #00acee;
            color: white;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
        }
        .chat-message.user {
            background-color: #262730;
        }
        .chat-message.bot {
            background-color: #1e1e1e;
        }
        .chat-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        .user .chat-avatar {
            background-color: #00acee;
        }
        .bot .chat-avatar {
            background-color: #4a4a4a;
        }
        .chat-content {
            flex-grow: 1;
        }
        .sidebar .markdown-text-container {
            background-color: #262730;
            border-radius: 0.5rem;
            padding: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Custom container for header
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #00acee; font-size: 3rem; margin-bottom: 1rem;'>AI Website Chat Assistant</h1>
        <p style='color: #666; font-size: 1.2rem;'>Chat with any website using AI</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with enhanced styling
with st.sidebar:
    st.markdown("""
        <div style='padding: 1rem; background-color: #262730; border-radius: 0.5rem; margin-bottom: 1rem;'>
            <h2 style='color: #00acee; margin-bottom: 1rem;'>⚙️ Configuration</h2>
        </div>
    """, unsafe_allow_html=True)
    
    website_url = st.text_input("Enter Website URL", placeholder="https://example.com")

# Main chat interface
if website_url is None or website_url == "":
    st.info("👋 Welcome! Please enter a website URL in the sidebar to start chatting.")
else:
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! How can I help you learn about this website?")
        ]
    if "vector_store" not in st.session_state:
        with st.spinner("Processing website content..."):
            st.session_state.vector_store = get_vectorstore(website_url)

    # Chat messages display
    for i, message in enumerate(st.session_state.chat_history):
        # Skip the initial greeting message if it's not the only message
        if i == 0 and len(st.session_state.chat_history) > 1:
            continue
            
        if isinstance(message, AIMessage):
            # Remove 'AI:' prefix if present
            content = message.content
            if content.startswith('AI:'):
                content = content[3:].strip()
                
            st.markdown(f"""
                <div class='chat-message bot'>
                    <div class='chat-avatar'>🤖</div>
                    <div class='chat-content'>{content}</div>
                </div>
            """, unsafe_allow_html=True)
        elif isinstance(message, HumanMessage):
            st.markdown(f"""
                <div class='chat-message user'>
                    <div class='chat-avatar'>👤</div>
                    <div class='chat-content'>{message.content}</div>
                </div>
            """, unsafe_allow_html=True)

    # User input
    user_input = st.chat_input("Ask me anything about the website...")
    
    if user_input is not None and user_input != "":
        with st.spinner("Thinking..."):
            response = get_response(user_input)
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=response))
            st.rerun()
