import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

# =====================================================================
# 1. INITIALIZATION & CONFIGURATION
# =====================================================================
st.set_page_config(page_title="Company AI Onboarding Hub", layout="wide")
st.title("🤝 Corporate Onboarding & Role Support Assistant")
st.caption("Learn your duties, explore company policies, or resolve role misalignments.")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "escalations" not in st.session_state:
    st.session_state.escalations = []

# SAFE API KEY LOADING: NO HARDCODING. 
deepseek_key = st.secrets.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")

if not deepseek_key:
    st.error("❌ DEEPSEEK_API_KEY missing! Go to your Streamlit Cloud Dashboard -> App Settings -> Secrets, and add: DEEPSEEK_API_KEY = 'your_key'")
    st.stop()

# Configure ChatOpenAI to route explicitly to DeepSeek
llm = ChatOpenAI(
    model="deepseek-chat", 
    openai_api_key=deepseek_key,
    openai_api_base="https://api.deepseek.com", # FIXED: Removed /v1 suffix for standard OpenAI SDK alignment
    temperature=0.1
)

# =====================================================================
# 2. TRAINING KNOWLEDGE BASE (RAG) Setup
# =====================================================================
@st.cache_resource
def setup_knowledge_base():
    onboarding_docs = [
        Document(page_content="Company Core Hours: 9 AM to 5 PM. Remote work requires prior team-lead approval.", metadata={"source": "HR-Policy"}),
        Document(page_content="Software Engineer Role: Responsibilities include writing clean Python code, participating in daily standups at 10 AM, and reviewing 2 pull requests daily.", metadata={"source": "Eng-Playbook"}),
        Document(page_content="Expense Reporting: Submit all monthly operational receipts via the internal portal by the 25th of each month.", metadata={"source": "Finance-Wiki"}),
    ]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(onboarding_docs)
    return vector_store.as_retriever(search_kwargs={"k": 2})

retriever = setup_knowledge_base()

# =====================================================================
# 3. UTILITY METHODS
# =====================================================================
def run_rag_search(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join([f"Source ({d.metadata['source']}): {d.page_content}" for d in docs])

def trigger_escalation(issue: str):
    log_entry = {"issue": issue, "status": "Pending HR Review"}
    st.session_state.escalations.append(log_entry)

# =====================================================================
# 4. SYSTEM PROMPT DESIGN
# =====================================================================
system_prompt = (
    "You are an empathetic, professional Corporate Onboarding Assistant.\n"
    "Your primary goal is to train new employees on their duties.\n\n"
    "You have access to the company handbook which contains information on HR-Policy, Eng-Playbook, and Finance-Wiki.\n"
    "CRITICAL RULE: If the employee states their tasks do not match their contract, expresses conflict with management "
    "expectations, or shows frustration about role responsibilities, you MUST tell them you are escalating this to HR "
    "and append the special trigger text '[TRIGGER_ESCALATION: <describe the user issue here>]' at the very end of your response.\n"
    "Always maintain a supportive, clear tone."
)

# =====================================================================
# 5. STREAMLIT GUI LAYOUT
# =====================================================================
col_chat, col_admin = st.columns([2, 1])

with col_chat:
    st.subheader("💬 Chat with your Onboarding Buddy")
    
    for msg in st.session_state.chat_history:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        with st.chat_message(role):
            st.write(msg.content)

    if user_input := st.chat_input("Ask a question about your role, or report a duty misalignment..."):
        with st.chat_message("user"):
            st.write(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Look up relevant documents first
                context_docs = run_rag_search(user_input)
                
                # 2. Build full conversational prompt context
                messages = [("system", system_prompt), ("system", f"Relevant Company Info:\n{context_docs}")]
                for m in st.session_state.chat_history[-4:]:  # keep last 4 context messages
                    role_type = "assistant" if isinstance(m, AIMessage) else "user"
                    messages.append((role_type, m.content))
                messages.append(("user", user_input))
                
                # 3. Call DeepSeek safely
                ai_response = llm.invoke(messages)
                output_text = ai_response.content
                
                # 4. FIXED: String extraction array split bug repaired completely
                if "[TRIGGER_ESCALATION:" in output_text:
                    try:
                        parts = output_text.split("[TRIGGER_ESCALATION:")
                        issue_details = parts[1].split("]")[0]
                        trigger_escalation(issue_details.strip())
                        
                        output_text = parts[0].strip()
                        output_text += "\n\n⚠️ *System Notification: This issue has been logged into the Management Panel.*"
                    except Exception:
                        trigger_escalation(user_input)
                
                st.write(output_text)
                
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=output_text))
        st.rerun()

with col_admin:
    st.subheader("🛡️ Management Control Panel")
    st.info("HR tracking panel view.")
    
    if not st.session_state.escalations:
        st.success("✅ No role misalignments reported yet.")
    else:
        for idx, ticket in enumerate(st.session_state.escalations, 1):
            with st.expander(f"⚠️ Ticket #{idx} - Misalignment Logged", expanded=True):
                st.error(f"**Reported Issue:** {ticket['issue']}")
                st.caption(f"**Current Status:** {ticket['status']}")
