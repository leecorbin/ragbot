import os
import glob
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# === Config ===
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DB_NAME = "vector_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 25
LLM_TEMPERATURE = 0.2

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY is required")
os.environ['OPENAI_API_KEY'] = api_key

text_loader_kwargs = {'encoding': 'utf-8'}

# === Helpers ===

def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc

def load_documents():
    folders = glob.glob("knowledge-base/*")
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs=text_loader_kwargs
        )
        docs = loader.load()
        documents.extend([add_metadata(doc, doc_type) for doc in docs])
    return documents

def build_or_load_vectorstore():
    embeddings = OpenAIEmbeddings()
    if os.path.exists(DB_NAME) and os.path.isdir(DB_NAME):
        print("Loading existing vectorstore...")
        return Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    else:
        print("Building new vectorstore...")
        docs = load_documents()
        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        print(f"Chunked into {len(chunks)} pieces")
        vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)
        return vs

def get_conversational_chain(vectorstore):
    llm = ChatOpenAI(temperature=LLM_TEMPERATURE, model_name=MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    
    prompt = PromptTemplate(
        input_variables=["question", "context", "chat_history"],
        template="""You are an expert assistant helping employees of Insurellm, an insurance technology company. 
Always answer based **only** on the provided context. Do not make up answers. 
If the context is unclear, do your best to provide a helpful and accurate response grounded in what you know from the documents. 
Avoid saying "I don't know" unless absolutely nothing in the context is relevant. 
Chat History: {chat_history} Context: {context} Question: {question} Helpful Answer:"""
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

def launch_app():
    vectorstore = build_or_load_vectorstore()
    chain = get_conversational_chain(vectorstore)

    def chat(question, history):
        return chain.invoke({"question": question})["answer"]

    gr.ChatInterface(chat, type="messages").launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_app()