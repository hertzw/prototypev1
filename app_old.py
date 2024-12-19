import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.corpus import stopwords
import os

# Baixar recursos necessários do NLTK
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('portuguese'))

# Carregar modelo de forma eficiente
@st.cache_resource
def carregar_modelo(modelo_selecionado):
    return SentenceTransformer(modelo_selecionado)

# Função para pré-processar os textos
def preprocessar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Záéíóúãõç\s]', '', texto)
    palavras = texto.split()
    palavras_filtradas = [palavra for palavra in palavras if palavra not in stop_words]
    return ' '.join(palavras_filtradas)

# Função principal para análise de similaridade
def analisar_similaridade(texto1, texto2, modelo_selecionado):
    modelo = carregar_modelo(modelo_selecionado)
    texto1 = preprocessar_texto(texto1)
    texto2 = preprocessar_texto(texto2)
    embeddings = modelo.encode([texto1, texto2], normalize_embeddings=True)
    similaridade = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similaridade

# Função auxiliar para limpar os campos
def limpar_campos():
    st.session_state.texto1 = ""
    st.session_state.texto2 = ""
    st.session_state.resultado = ""

# Inicialização segura do st.session_state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "resultado" not in st.session_state:
    st.session_state.resultado = ""
if "texto1" not in st.session_state:
    st.session_state.texto1 = ""
if "texto2" not in st.session_state:
    st.session_state.texto2 = ""

# Tela de Login
def login():
    st.title("Autenticação")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        # Use variáveis de ambiente ou um método seguro para autenticação
        if username == os.getenv("USER") and password == os.getenv("PASSWORD"):
            st.session_state.logged_in = True
            st.success("Login bem-sucedido!")
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos.")

# Página principal com login obrigatório
if not st.session_state.logged_in:
    login()
else:
    # Configuração da página principal
    st.set_page_config(page_title="Calculadora de Similaridade", layout="wide")

    # Menu lateral
    st.sidebar.title("Menu")
    pagina = st.sidebar.radio("Navegue entre as páginas:", ["Calculadora", "Modelos Disponíveis"])

    # Página Principal
    if pagina == "Calculadora":
        st.title("Calculadora de Similaridade entre Textos")

        # Seleção do Modelo
        modelo_selecionado = st.sidebar.selectbox(
            "Selecione o Modelo:",
            [
                "paraphrase-multilingual-MiniLM-L12-v2",
                "sentence-transformers/LaBSE",
                "all-MiniLM-L6-v2",
                "bert-base-uncased"
            ],
            index=0
        )

        # Layout de entrada
        col1, col2 = st.columns(2)

        with col1:
            st.text_area("Texto 1", key="texto1")

        with col2:
            st.text_area("Texto 2", key="texto2")

        # Botões
        col3, col4 = st.columns([1, 1])

        with col3:
            st.button("Limpar Tudo", on_click=limpar_campos)

        with col4:
            processar = st.button("Processar")

        # Processamento da similaridade
        if processar:
            if st.session_state.texto1.strip() and st.session_state.texto2.strip():
                with st.spinner("Calculando similaridade..."):
                    similaridade = analisar_similaridade(
                        st.session_state.texto1, st.session_state.texto2, modelo_selecionado
                    )
                    percentual_similaridade = similaridade * 100
                    st.session_state.resultado = {
                        "modelo": modelo_selecionado,
                        "similaridade": similaridade,
                        "percentual": percentual_similaridade,
                    }
            else:
                st.warning("Por favor, preencha ambos os campos de texto antes de processar.")

        # Exibir resultados
        if st.session_state.resultado:
            st.subheader("Resultados:")
            st.write(f"**Modelo Utilizado:** `{st.session_state.resultado['modelo']}`")
            st.write(f"**Similaridade entre os textos:** {st.session_state.resultado['similaridade']:.2f}")
            st.write(f"**Similaridade em percentual:** {st.session_state.resultado['percentual']:.2f}%")

    # Página de Modelos Disponíveis
    elif pagina == "Modelos Disponíveis":
        st.title("Modelos Disponíveis")
        st.write("""
        **1. paraphrase-multilingual-MiniLM-L12-v2**: Modelo leve e rápido para múltiplos idiomas.\n
        **2. sentence-transformers/LaBSE**: Modelo robusto para embeddings multilíngues.\n
        **3. all-MiniLM-L6-v2**: Compacto e eficiente para tarefas em tempo real.\n
        **4. bert-base-uncased**: Modelo padrão de BERT, utilizado em várias tarefas de NLP.\n
        """)

    # Logout
    st.sidebar.write("---")
    if st.sidebar.button("Logout"):
        # Reiniciar apenas variáveis que não são associadas diretamente a widgets
        st.session_state.logged_in = False
        st.session_state.resultado = None
        st.rerun()
