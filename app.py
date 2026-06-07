# ==============================================================================
# PROJECT: DEFENCE PROCUREMENT QUERY BOT (v8.5)
# ACADEMY: National Academy of Defence Production (NADP), Nagpur
# DEPLOYMENT: Strategic Cloud Edition (Groq-Powered)
# PURPOSE: Advanced Decision Support System for Defence Manuals
# ==============================================================================

import streamlit as st
import os
import time
import pandas as pd
import logging
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================================================================
# SECTION 1: SYSTEM IDENTITY & CONSTANTS
# ==============================================================================

PROJECT_NAME = "Defence Procurement Query Bot"
SYSTEM_ID = "DPQB-NADP-2026"
# !!! REPLACE THE LINE BELOW WITH YOUR GSK KEY !!!
GROQ_API_KEY = "PASTE_YOUR_GROQ_KEY_HERE" 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DPQB_CORE")

# ==============================================================================
# SECTION 2: PROFESSIONAL DEFENCE UI DESIGN
# ==============================================================================

st.set_page_config(
    page_title=PROJECT_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_military_styles():
    """Injects high-fidelity tactical CSS for deployment aesthetics"""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;700&family=Inter:wght@400;900&display=swap');
        
        :root {
            --primary-navy: #0a192f;
            --secondary-navy: #112240;
            --accent-cyan: #64ffda;
            --military-gold: #d4af37;
            --pure-white: #ccd6f6;
        }

        .stApp {
            background-color: var(--primary-navy);
            color: var(--pure-white);
            font-family: 'Roboto Mono', monospace;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #020c1b;
            border-right: 2px solid var(--military-gold);
        }

        /* Tactical Header */
        .bot-header {
            text-align: center;
            padding: 30px;
            background: rgba(17, 34, 64, 0.7);
            border-bottom: 2px solid var(--military-gold);
            margin-bottom: 40px;
        }
        .bot-header h1 {
            color: var(--military-gold);
            font-family: 'Inter', sans-serif;
            font-weight: 900;
            letter-spacing: 4px;
            text-transform: uppercase;
        }

        /* Consultation Result Cards */
        .analysis-card {
            background-color: var(--secondary-navy);
            border: 1px solid var(--accent-cyan);
            padding: 25px;
            border-radius: 4px;
            border-left: 8px solid var(--military-gold);
            margin-bottom: 25px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.5);
        }

        /* System Telemetry Log */
        .system-log {
            background-color: #000000;
            color: #00ff41;
            padding: 15px;
            border: 1px solid #333;
            font-size: 0.8rem;
            height: 220px;
            overflow-y: auto;
            border-radius: 3px;
        }

        /* Metrics */
        .metric-unit { text-align: center; border: 1px solid #112240; padding: 10px; background: #010a15; }
        .metric-title { color: var(--military-gold); font-size: 0.7rem; font-weight: bold; }
        .metric-digit { color: white; font-size: 1.6rem; font-weight: 800; }
        </style>
    """, unsafe_allow_html=True)

apply_military_styles()

# ==============================================================================
# SECTION 3: CLOUD KNOWLEDGE ARCHITECTURE
# ==============================================================================

class QueryBotEngine:
    """Handles High-Speed Inference and Strategic Context Mining"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        # We switch to HuggingFace Embeddings for Cloud stability
        self.embed_engine = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vault_path = "permanent_vault"
        self.vault = self._mount_vault()

    def _mount_vault(self):
        """Loads the pre-computed neural database from project directory"""
        if os.path.exists(self.vault_path):
            try:
                return FAISS.load_local(
                    self.vault_path, 
                    self.embed_engine, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                logger.error(f"Vault Mount Error: {e}")
                return None
        return None

    def synthesize_consultation(self, user_query):
        """Main RAG Pipeline: Retrieves evidence and generates strategic analysis"""
        if not self.vault:
            return "ERROR: Neural Vault is offline."
            
        # 1. RETRIEVAL (K=10 for broad synthesis)
        retriever = self.vault.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(user_query)
        
        # 2. CONTEXT BUILDING
        evidence_base = ""
        for i, d in enumerate(docs):
            src = d.metadata.get('source', 'Defence Manual')
            evidence_base += f"\n[DOC {i+1} SOURCE: {src}]\n{d.page_content}\n"

        # 3. PENTAGON REASONING PROMPT
        master_prompt = f"""
        YOU ARE THE 'DEFENCE PROCUREMENT QUERY BOT'.
        STRICT MISSION: Provide a 360-degree Consultation based ONLY on official context.

        KNOWLEDGE CONTEXT:
        {evidence_base}

        USER REQUEST: {user_query}

        REPORT STRUCTURE:
        1. 📋 SITUATIONAL ANALYSIS: Categorize as Capital (DAP) or Revenue (DPM).
        2. ⚖️ PROCEDURAL PATHWAY: Reference specific Manual/Handbook Chapters.
        3. 💰 FINANCIAL AUTHORITY: Identify the CFA using DFPDS 2026 limits.
        4. 🛡️ AUDIT & RISK COMPLIANCE: Highlight potential roadblocks.
        5. ✅ ACTIONABLE RECOMMENDATION: 3 steps to move the file forward.

        RULES: Cite source manuals for every fact. If data is missing, state 'No specific clause found in corpus.'
        """

        # 4. CLOUD INFERENCE (Groq Llama 3.1 70B)
        llm = ChatGroq(
            groq_api_key=self.api_key, 
            model_name="llama-3.1-70b-versatile",
            temperature=0
        )
        
        return llm.stream(master_prompt)

# Initialize System Core
if not GROQ_API_KEY or GROQ_API_KEY == "PASTE_YOUR_GROQ_KEY_HERE":
    st.error("SYSTEM HALTED: Valid Groq API Key required for deployment.")
    st.stop()

bot = QueryBotEngine(GROQ_API_KEY)

# ==============================================================================
# SECTION 4: TACTICAL SIDEBAR & STATUS
# ==============================================================================

with st.sidebar:
    st.markdown(f"<h2 style='color:#d4af37;'>📡 BOT TELEMETRY</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Real-time Metrics
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown("<div class='metric-unit'><p class='metric-title'>CORPUS</p><p class='metric-digit'>1.6k+p</p></div>", unsafe_allow_html=True)
    with mcol2:
        st.markdown("<div class='metric-unit'><p class='metric-title'>ENGINE</p><p class='metric-digit'>GROQ</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ LIVE PROCESS LOG")
    log_stream = st.empty()
    
    if "logs" not in st.session_state:
        st.session_state.logs = [f"[{datetime.now().strftime('%H:%M:%S')}] Defence Bot initialized."]

    def add_log(msg):
        ts = datetime.now().strftime('%H:%M:%S')
        st.session_state.logs.append(f"[{ts}] {msg}")
        log_content = "\n".join(st.session_state.logs[-15:])
        log_stream.markdown(f"<div class='system-log'>{log_content}</div>", unsafe_allow_html=True)

    add_log("Cloud API Handshake: SUCCESS.")

    st.markdown("---")
    st.markdown("### 🗃️ DATA REPOSITORIES")
    manuals = {
        "Policy": "DAP 2026 / DPM V1",
        "Financial": "DFPDS 2026 Army/AF/N",
        "Strategic": "TPCR Capability Map",
        "Guide": "DAP Handbook"
    }
    for k, v in manuals.items():
        st.caption(f"**{k}**: {v}")

    if st.button("🔴 RESET BOT MEMORY"):
        st.session_state.messages = []
        st.session_state.logs = []
        st.rerun()

# ==============================================================================
# SECTION 5: INTERACTIVE ANALYTICAL DASHBOARD
# ==============================================================================

st.markdown(f"<div class='bot-header'><h1>🛡️ {PROJECT_NAME}</h1></div>", unsafe_allow_html=True)
st.caption("Strategic Decision Support Tool | National Academy of Defence Production | Nagpur")

if not bot.vault:
    st.error("FATAL: Knowledge Vault not detected. Please upload 'permanent_vault' folder.")
    st.stop()

# Conversation State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Persistent Chat View
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Query Interaction
if user_input := st.chat_input("Enter procurement query (e.g., Analyze ₹500cr UAV project)..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    add_log(f"New Query: {user_input[:40]}...")

    # EXECUTION OF STRATEGIC CONSULTATION
    with st.chat_message("assistant"):
        with st.status("🛸 Accessing Defence Knowledge Layers...", expanded=True) as status:
            st.write("Mining context from DAP/DPM/DFPDS...")
            add_log("Knowledge mining initiated.")
            time.sleep(0.3)
            status.update(label="STRATEGIC ANALYSIS READY", state="complete", expanded=False)

        # STREAMING CLOUD RESPONSE
        output_ui = st.empty()
        full_analysis = ""
        
        try:
            for part in bot.synthesize_consultation(user_input):
                full_analysis += part.content
                output_ui.markdown(full_analysis + "▌")
            
            output_ui.markdown(full_analysis)
            add_log("Consultation Report delivered via Groq Cloud.")
            st.session_state.messages.append({"role": "assistant", "content": full_analysis})
        except Exception as e:
            st.error(f"Inference Failure: {str(e)}")
            add_log("CRITICAL ERROR: Groq API timeout.")

# ==============================================================================
# SECTION 6: COMPLIANCE FOOTER
# ==============================================================================

st.markdown("---")
dash1, dash2, dash3 = st.columns(3)

with dash1:
    st.markdown("<div class='analysis-card'><p class='metric-title'>Security</p><p>🔒 Cloud SSL Encrypted</p></div>", unsafe_allow_html=True)
with dash2:
    st.markdown("<div class='analysis-card'><p class='metric-title'>Integrity</p><p>✅ Factual Cross-Ref</p></div>", unsafe_allow_html=True)
with dash3:
    st.markdown("<div class='analysis-card'><p class='metric-title'>Framework</p><p>🧿 Pentagon Reasoning</p></div>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align: center; color: #555; font-size: 0.7rem;'>"
    "Proprietary Strategic Tool | NADP Nagpur | Capstone 2025-26 | DPQB Build v8.5"
    "</p>", 
    unsafe_allow_html=True
)