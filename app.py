from dotenv import load_dotenv
import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import tempfile
import urllib.request
import langchain
from langchain.chat_models import ChatOpenAI

langchain.verbose = False

# Carrega as variáveis de ambiente
load_dotenv()  # Isso carregará as variáveis de ambiente do arquivo .env no diretório atual

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def load_text_from_url(url):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(urllib.request.urlopen(url).read())
        temp_file_path = temp_file.name

    text = ""
    with open(temp_file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    return text

def process_text(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=500, length_function=len)
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=os.environ.get("OPENAI_API_KEY"))
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

def message(content, is_user=False, key=None, logo_url=None):
    col1, col2 = st.columns([1, 5])
    if is_user:
        # Mensagem do usuário à direita
        with col2:
            st.text_area("", value=content, height=100, key=key, help="Pergunta do usuário")
    else:
        # Logo e mensagem do assistente à esquerda
        with col1:
            if logo_url:
                st.image(logo_url, width=30)
        with col2:
            st.text_area("", value=content, height=100, key=key, help="Resposta do assistente", disabled=True)


def main():

    logo_url = 'https://github.com/pedrosale/falcon_test/raw/a7248c8951827efd997b927d7a4d4c4c200c1996/logo_det3.png'

    st.title("Detran + OpenAI 💬 CTB")
    st.image(logo_url, width=45)  # Ajuste a largura conforme necessário
    st.markdown('**Esta versão contém:**  \nA) Modelo gpt-3.5-turbo-0125;  \nB) Conjunto de dados pré-carregados;  \nC) ["Retrieval Augmented Generation"](https://python.langchain.com/docs/use_cases/question_answering/) a partir dos dados carregados.')

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    file_path1 = "https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt"
    text1 = load_text_from_url(file_path1)

    knowledgeBase = process_text(text1)

    # Substitua o campo de entrada de texto e o botão por um formulário
    with st.form(key='query_form'):
        query = st.text_input('Me pergunte...', key='query_input')
        send_button = st.form_submit_button('Enviar')

    if send_button and query:
        docs = knowledgeBase.similarity_search(query)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=1, max_tokens=1500,openai_api_key=os.environ.get("OPENAI_API_KEY"))
        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cost:
            response = chain.invoke(input={"question": query, "input_documents": docs})

        st.session_state.chat_history.append({"pergunta": query, "resposta": response["output_text"]})

    # Exibe o histórico de conversas
    for chat in st.session_state.chat_history:
        message("Pergunta: " + chat["pergunta"], is_user=True)
        message("Resposta: " + chat["resposta"], is_user=False, logo_url=logo_url)
        st.text("---")  # Linha separadora

if __name__ == "__main__":
  main()
