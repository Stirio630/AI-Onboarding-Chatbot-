import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

# 1. Paste your real DeepSeek key here or use Streamlit secrets
if "DEEPSEEK_API_KEY" not in os.environ:
    os.environ["DEEPSEEK_API_KEY"] = "sk-19c9844c29984c8481c97e0c85c2427d"

# 2. Configure ChatOpenAI to target the DeepSeek endpoint and model
llm = ChatOpenAI(
    model="deepseek-chat", # Uses DeepSeek-V3 or DeepSeek-R1 depending on your needs
    openai_api_key=os.environ["DEEPSEEK_API_KEY"],
    openai_api_base="https://deepseek.com",
    temperature=0.2
)

# Note: The rest of your agent and Streamlit code remains exactly the same!

# =====================================================================
# 2. TRAINING KNOWLEDGE BASE (RAG) Setup
# =====================================================================
@st.cache_resource
def setup_knowledge_base():
    """Simulates loading company onboarding documents into a vector database."""
    onboarding_docs = [
        Document(page_content="Company Core Hours: 9 AM to 5 PM. Remote work requires prior team-lead approval.", metadata={"source": "HR-Policy"}),
        Document(page_content="Software Engineer Role: Responsibilities include writing clean Python code, participating in daily standups at 10 AM, and reviewing 2 pull requests daily.", metadata={"source": "Eng-Playbook"}),
        Document(page_content="Expense Reporting: Submit all monthly operational receipts via the internal portal by the 25th of each month.", metadata={"source": "Finance-Wiki"}),
    ]
    embeddings = OpenAIEmbeddings()
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(onboarding_docs)
    return vector_store.as_retriever(search_kwargs={"k": 2})

retriever = setup_knowledge_base()

# =====================================================================
# 3. CUSTOM LANGCHAIN TOOLS (RAG & Escalation)
# =====================================================================
@tool
def query_company_handbook(query: str) -> str:
    """Useful when you need to answer questions about company rules, policies, schedules, roles, and duties."""
    docs = retriever.invoke(query)
    return "\n\n".join([f"Source ({d.metadata['source']}): {d.page_content}" for d in docs])

@tool
def escalate_to_management(employee_issue: str) -> str:
    """Useful ONLY when an employee expresses explicit misalignment, confusion, or conflict regarding their role, duties, or management expectations."""
    # In production, this would trigger an email via SendGrid or log to a database
    log_entry = {"issue": employee_issue, "status": "Pending HR Review"}
    st.session_state.escalations.append(log_entry)
    return "SUCCESS: This critical role misalignment has been flagged and escalated to Management/HR. A representative will schedule a meeting with you shortly."

tools = [query_company_handbook, escalate_to_management]

# =====================================================================
# 4. AGENT ASSEMBLY
# =====================================================================
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an empathetic, professional Corporate Onboarding Assistant.\n"
        "Your primary goal is to train new employees on their duties using the 'query_company_handbook' tool.\n"
        "CRITICAL RULE: If the employee states their assigned tasks do not match their contract, "
        "expresses conflict with management expectations, or shows deep frustration about their role responsibilities, "
        "you MUST immediately use the 'escalate_to_management' tool to protect their onboarding experience. "
        "Be supportive and clear about the escalation process."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# =====================================================================
# 5. STREAMLIT GUI LAYOUT
# =====================================================================
col_chat, col_admin = st.columns([2, 1])

with col_chat:
    st.subheader("💬 Chat with your Onboarding Buddy")
    
    # Display historical messages
    for msg in st.session_state.chat_history:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        with st.chat_message(role):
            st.write(msg.content)

    # Capture User input
    if user_input := st.chat_input("Ask a question about your role, or report a duty misalignment..."):
        with st.chat_message("user"):
            st.write(user_input)
            
        # Run agent loop
        with st.chat_message("assistant"):
            with st.spinner("Processing request..."):
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                output_text = response["output"]
                st.write(output_text)
                
        # Append to persistent session memory
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=output_text))
        st.rerun()

with col_admin:
    st.subheader("🛡️ Management Control Panel")
    st.info("This section simulates what HR/Management sees when a ticket is created.")
    
    if not st.session_state.escalations:
        st.success("✅ No role misalignments reported yet.")
    else:
        for idx, ticket in enumerate(st.session_state.escalations, 1):
            with st.expander(f"⚠️ Ticket #{idx} - Misalignment Logged", expanded=True):
                st.error(f"**Reported Issue:** {ticket['issue']}")
                st.caption(f"**Current Status:** {ticket['status']}")
