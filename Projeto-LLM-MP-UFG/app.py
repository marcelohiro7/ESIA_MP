import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import os

load_dotenv()

# --- Configuração da página ---
_icon = Image.open(Path(__file__).parent / "mp.png")
st.set_page_config(page_title="Atendimento e Suporte de TI do Ministério Público", page_icon=_icon)
st.title("Atendimento e Suporte de TI - Ministério Público")

# --- Configurações ---
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, max_retries=2)

DOCS_PATH = str(Path(__file__).parent / "documentos")
FAISS_PATH = str(Path(__file__).parent / "index_faiss")

SYSTEM_PROMPT = """Você é um assistente virtual prestativo de TI do Ministério Público.
Use o contexto recuperado para responder à pergunta.
Se não souber a resposta, diga que não sabe com certeza.
Se for uma dúvida comum, sugira uma solução alternativa possível.
Mantenha a resposta concisa. Responda em português."""


# --- Funções ---
def carregar_documentos(pasta):
    """Carrega documentos PDF da pasta informada."""
    docs_path = Path(pasta)
    documentos = []
    for arq in docs_path.iterdir():
        ext = arq.suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyMuPDFLoader(str(arq))
                pages = loader.load()
                documentos.append("\n".join([p.page_content for p in pages]))
            elif ext == ".txt":
                documentos.append(arq.read_text(encoding="utf-8"))
        except Exception:
            pass
    return documentos


def configurar_retriever(pasta):
    """Configura o retriever FAISS a partir dos documentos."""
    # Tenta carregar índice existente
    faiss_index = Path(FAISS_PATH) / "index.faiss"
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    if faiss_index.exists():
        vectorstore = FAISS.load_local(
            FAISS_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        documentos = carregar_documentos(pasta)
        if not documentos:
            st.error("Nenhum documento encontrado na pasta de documentos.")
            st.stop()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = []
        for doc in documentos:
            chunks.extend(splitter.split_text(doc))
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        vectorstore.save_local(FAISS_PATH)

    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4})


def configurar_rag(llm, retriever):
    """Monta a chain RAG com histórico de conversa."""
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the follow-up question, reformulate it as a standalone question. Do NOT answer, just reformulate if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "Question: {input}"),
    ])
    history_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "Pergunta: {input}\n\nContexto: {context}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_retriever, qa_chain)


def chat(rag_chain, pergunta, historico):
    """Envia pergunta para a chain e retorna a resposta."""
    historico.append(HumanMessage(content=pergunta))
    resposta = rag_chain.invoke({"input": pergunta, "chat_history": historico})
    res = resposta["answer"]
    if "</think>" in res:
        res = res.split("</think>")[-1].strip()
    historico.append(AIMessage(content=res))
    return res


# --- Inicialização de estado ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá! Sou o assistente de TI do MP. Como posso ajudar?")
    ]

if "rag_chain" not in st.session_state:
    retriever = configurar_retriever(DOCS_PATH)
    st.session_state.rag_chain = configurar_rag(llm, retriever)

# --- Exibir histórico ---
for msg in st.session_state.chat_history:
    if isinstance(msg, AIMessage):
        with st.chat_message("AI", avatar="🏛️"):
            st.write(msg.content)
    else:
        with st.chat_message("Human", avatar="👤"):
            st.write(msg.content)

# --- Input do usuário ---
pergunta = st.chat_input("Digite sua mensagem...")
if pergunta:
    with st.chat_message("Human", avatar="👤"):
        st.write(pergunta)
    with st.chat_message("AI", avatar="🏛️"):
        res = chat(st.session_state.rag_chain, pergunta, st.session_state.chat_history)
        st.write(res)
