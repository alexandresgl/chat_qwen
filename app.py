import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Carrega a GROQ_API_KEY exclusivamente do arquivo .env
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Configuração inicial da página
st.set_page_config(
    page_title="Qwen AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Customizado com contraste travado para evitar falhas de leitura
st.markdown(
    """
<style>
    /* Fundo principal e cabeçalho superior */
    .stApp, [data-testid="stHeader"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc !important;
    }

    /* Estilização da barra lateral */
    [data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid #1e293b;
    }

    /* Garante textos legíveis na barra lateral */
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    /* Bolhas de mensagem do chat */
    [data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.85) !important;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
    }

    /* Força cor clara em todos os textos dentro das mensagens */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        color: #f8fafc !important;
    }

    /* Área inferior do campo de digitação */
    [data-testid="stBottom"] > div {
        background-color: #0f172a !important;
    }

    /* Título com gradiente */
    .titulo-gradiente {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitulo {
        color: #94a3b8 !important;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Botão estilizado */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #334155 !important;
        background-color: #1e293b !important;
        color: #f8fafc !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Busca os modelos disponíveis direto na conta Groq usando a chave do .env
@st.cache_data(ttl=300)
def listar_modelos_disponiveis(chave_api):
    if not chave_api:
        return ["qwen/qwen3-32b", "qwen-qwq-32b"]
    try:
        client = Groq(api_key=chave_api)
        lista = client.models.list()
        todos_modelos = [m.id for m in lista.data]
        modelos_qwen = sorted([m for m in todos_modelos if "qwen" in m.lower()])
        return modelos_qwen if modelos_qwen else sorted(todos_modelos)
    except Exception:
        return ["qwen/qwen3-32b", "qwen-qwq-32b"]


# --- BARRA LATERAL (CONFIGURAÇÕES) ---
with st.sidebar:
    st.markdown("### ⚙️ Configurações do Modelo")

    # Status discreto da chave (sem expor o valor na tela)
    if API_KEY:
        st.success("🔒 API Key autenticada via `.env`")
    else:
        st.error("⚠️ `GROQ_API_KEY` não encontrada no `.env`")

    opcoes_modelos = listar_modelos_disponiveis(API_KEY)

    modelo_selecionado = st.selectbox(
        "Modelo Qwen",
        options=opcoes_modelos,
        index=0,
        help="Modelos ativos detectados automaticamente na sua conta Groq.",
    )

    temperatura = st.slider(
        "Criatividade (Temperature)", min_value=0.0, max_value=1.5, value=0.7, step=0.1
    )

    system_prompt = st.text_area(
        "Instrução do Sistema (Persona)",
        value="Você é um assistente de inteligência artificial prestativo, claro e objetivo. Responda sempre em português do Brasil usando formatação Markdown quando ajudar na leitura.",
        height=120,
    )

    st.divider()

    if st.button("🗑️ Limpar Conversa"):
        st.session_state.mensagens = []
        st.rerun()

    st.caption("Desenvolvido em Python + Streamlit + Groq API (Qwen)")

# --- CABEÇALHO PRINCIPAL ---
st.markdown(
    '<h1 class="titulo-gradiente">Qwen Studio Chat</h1>', unsafe_allow_html=True
)
st.markdown(
    f'<p class="subtitulo">Conectado via Groq API • Modelo ativo: <b>{modelo_selecionado}</b></p>',
    unsafe_allow_html=True,
)

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if len(st.session_state.mensagens) == 0:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            "Olá! Sou o assistente rodando o modelo **Qwen**. Como posso ajudar você hoje?"
        )

for msg in st.session_state.mensagens:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


def gerar_resposta_stream(client, mensagens_api, modelo, temp):
    stream = client.chat.completions.create(
        model=modelo,
        messages=mensagens_api,
        temperature=temp,
        max_tokens=2048,
        stream=True,
    )
    for chunk in stream:
        conteudo = chunk.choices[0].delta.content
        if conteudo:
            yield conteudo


# --- CAMPO DE ENTRADA DO USUÁRIO ---
if prompt_usuario := st.chat_input("Digite sua mensagem para o Qwen..."):
    if not API_KEY:
        st.error(
            "⚠️ Configure sua GROQ_API_KEY no arquivo .env antes de enviar mensagens."
        )
        st.stop()

    st.session_state.mensagens.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt_usuario)

    mensagens_para_api = [
        {"role": "system", "content": system_prompt}
    ] + st.session_state.mensagens

    with st.chat_message("assistant", avatar="🤖"):
        try:
            client = Groq(api_key=API_KEY)
            resposta_completa = st.write_stream(
                gerar_resposta_stream(
                    client, mensagens_para_api, modelo_selecionado, temperatura
                )
            )
            st.session_state.mensagens.append(
                {"role": "assistant", "content": resposta_completa}
            )
        except Exception as e:
            st.error(f"Erro ao comunicar com a API da Groq: {e}")
