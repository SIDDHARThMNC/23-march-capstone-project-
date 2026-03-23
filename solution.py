"""
=============================================================================
  COMPLETE SOLUTION — Single File
  Question 1: End-to-End RAG + Agent System       (Ed-Tech Assistant)
  Question 2: Multi-Agent Workflow with LangGraph  (Fintech Support)
=============================================================================

DEPENDENCIES (pip install):
    openai langchain langchain-openai langchain-community
    langgraph faiss-cpu pypdf tiktoken python-dotenv

Set your key:  export OPENAI_API_KEY="sk-..."
=============================================================================
"""

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# ─────────────────────────────────────────────────────────────────────────────
#  SHARED LLM
# ─────────────────────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,              # low temp → less hallucination (Reliability #1)
    openai_api_key=OPENAI_API_KEY,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
)

# =============================================================================
#  QUESTION 1 — RAG + AGENT SYSTEM
# =============================================================================

# ── 1A. RAG PIPELINE ──────────────────────────────────────────────────────────

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ── Mock documents (replace with real PDF loader in production) ───────────────
# In production: use PyPDFLoader + DirectoryLoader to load real PDFs
MOCK_DOCS = [
    Document(page_content="""
        Refund Policy: Students are eligible for a full refund within 7 days of enrollment,
        provided they have completed less than 20% of the course. After 7 days or 20% completion,
        no refund will be issued. To request a refund, contact support@edtech.com.
    """, metadata={"source": "policies.pdf", "page": 1}),

    Document(page_content="""
        Assignment Deadlines Policy: All assignments must be submitted by 11:59 PM on the due date.
        Late submissions incur a 10% penalty per day. Extensions can be requested 48 hours in advance
        by emailing your instructor. No extensions granted after the deadline has passed.
    """, metadata={"source": "policies.pdf", "page": 2}),

    Document(page_content="""
        Python Basics — Lecture 3: Functions and Scope.
        A function is defined using the 'def' keyword. Variables inside a function are local by default.
        Use the 'global' keyword to modify a global variable inside a function.
        Lambda functions are anonymous single-expression functions: lambda x: x * 2
    """, metadata={"source": "python_basics_lecture3.pdf", "page": 1}),

    Document(page_content="""
        Data Science 101 — Lecture 5: Pandas DataFrames.
        A DataFrame is a 2D labeled data structure. Use pd.read_csv() to load data.
        Key operations: df.head(), df.describe(), df.groupby(), df.merge().
        Always check for null values using df.isnull().sum() before analysis.
    """, metadata={"source": "ds101_lecture5.pdf", "page": 1}),

    Document(page_content="""
        Enrollment FAQ: To enroll in a course, log in to the portal and click 'Enroll Now'.
        Course access is granted immediately after payment. You can enroll in multiple courses.
        Enrollment confirmation is sent to your registered email within 10 minutes.
    """, metadata={"source": "faq.txt", "page": 1}),
]


def build_rag_pipeline(documents: list) -> FAISS:
    """
    Ingestion Pipeline:
    1. Chunk documents using RecursiveCharacterTextSplitter
       - Why recursive? Tries paragraph → sentence → word splits in order,
         preserving semantic coherence better than fixed-size splitting.
       - chunk_size=512: enough context per chunk, small enough for precision.
       - chunk_overlap=64: ~12% overlap prevents losing context at boundaries.
    2. Embed with text-embedding-3-small (strong quality, low cost).
    3. Store in FAISS (lightweight, local, fast ANN search — ideal for prototype).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[RAG] {len(documents)} docs → {len(chunks)} chunks after splitting")

    vector_store = FAISS.from_documents(chunks, embeddings)
    print("[RAG] Vector store built with FAISS")
    return vector_store


def retrieve_context(vector_store: FAISS, query: str, top_k: int = 4) -> str:
    """
    Retrieval Logic:
    - Similarity search returns top_k most relevant chunks.
    - Source metadata is appended so the LLM knows where info came from.
    - This grounding reduces hallucination (Reliability #2 — source attribution).
    """
    results = vector_store.similarity_search(query, k=top_k)
    context_parts = []
    for i, doc in enumerate(results):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content.strip()}")
    return "\n\n".join(context_parts)

# ── 1B. TOOLS (for student-specific data) ────────────────────────────────────

from langchain.tools import tool

# Mock student database
MOCK_STUDENTS = {
    "S001": {
        "name": "Rahul Sharma",
        "enrolled_courses": ["Python Basics", "Data Science 101"],
        "progress": {"Python Basics": 75, "Data Science 101": 40},
        "status": "active",
        "fees_due": 0,
    },
    "S002": {
        "name": "Priya Mehta",
        "enrolled_courses": ["Web Development"],
        "progress": {"Web Development": 90},
        "status": "active",
        "fees_due": 2500,
    },
    "S003": {
        "name": "Amit Kumar",
        "enrolled_courses": [],
        "progress": {},
        "status": "inactive",
        "fees_due": 0,
    },
}

MOCK_DEADLINES = {
    "python basics":    {"Assignment 1": "2026-04-01", "Assignment 2": "2026-04-15", "Final Project": "2026-05-01"},
    "data science 101": {"Assignment 1": "2026-03-30", "Midterm": "2026-04-10"},
    "web development":  {"Assignment 1": "2026-04-05", "Final Project": "2026-04-28"},
}


@tool
def get_student_status(student_id: str) -> str:
    """
    Fetch enrollment status, enrolled courses, and progress for a student.
    Use when the user asks about their enrollment, courses, or progress.
    Input: student_id (e.g. 'S001')
    """
    s = MOCK_STUDENTS.get(student_id.upper())
    if not s:
        return f"No student found with ID '{student_id}'."
    progress = "\n".join(f"  {c}: {p}%" for c, p in s["progress"].items()) or "  None"
    return (
        f"Name: {s['name']} | Status: {s['status'].capitalize()}\n"
        f"Courses: {', '.join(s['enrolled_courses']) or 'None'}\n"
        f"Progress:\n{progress}\n"
        f"Fees Due: ₹{s['fees_due']}"
    )


@tool
def get_assignment_deadlines(course_name: str) -> str:
    """
    Fetch assignment deadlines for a specific course.
    Use when the user asks about deadlines or due dates.
    Input: course_name (e.g. 'Python Basics')
    """
    key = course_name.lower()
    if key not in MOCK_DEADLINES:
        return f"No deadlines found for '{course_name}'. Available: {', '.join(MOCK_DEADLINES)}"
    lines = "\n".join(f"  {task}: {date}" for task, date in MOCK_DEADLINES[key].items())
    return f"Deadlines for '{course_name}':\n{lines}"


@tool
def check_refund_eligibility(student_id: str, course_name: str) -> str:
    """
    Check if a student is eligible for a refund on a course.
    Use when the user asks about refunds.
    Input: student_id, course_name
    """
    s = MOCK_STUDENTS.get(student_id.upper())
    if not s:
        return f"Student ID '{student_id}' not found."
    if course_name not in s["enrolled_courses"]:
        return f"{s['name']} is not enrolled in '{course_name}'."
    pct = s["progress"].get(course_name, 0)
    if pct < 20:
        return f"✅ ELIGIBLE for refund on '{course_name}'. Progress: {pct}% (threshold <20%)."
    return f"❌ NOT eligible for refund on '{course_name}'. Progress: {pct}% (threshold <20%)."


Q1_TOOLS = [get_student_status, get_assignment_deadlines, check_refund_eligibility]

# ── 1C. AGENT WITH RAG + TOOL DECISION LOGIC ─────────────────────────────────

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import SystemMessage

# System prompt — instructs agent when to use RAG vs tools
# Reliability #1: temperature=0 (set on LLM above)
# Reliability #2: explicit grounding instruction — "only use provided context"
# Reliability #3: source citation requirement
SYSTEM_PROMPT = """You are an intelligent assistant for an ed-tech platform.

You have access to:
1. A knowledge base (retrieved context) with course policies, lecture notes, and FAQs.
2. Tools to fetch real-time student-specific data.

Decision rules:
- For general questions about policies, course content, or FAQs → use the retrieved CONTEXT below.
- For student-specific questions (enrollment, progress, deadlines, refunds) → use the appropriate TOOL.
- NEVER make up information. If context is insufficient and no tool applies, say "I don't have that information."
- Always cite the source when answering from context (e.g., "According to policies.pdf...").

Retrieved Context:
{context}
"""

def create_edtech_agent(vector_store: FAISS) -> AgentExecutor:
    """
    Creates a tool-calling agent that:
    - Receives retrieved RAG context injected into the system prompt
    - Decides whether to answer from context OR call a tool
    - Uses create_tool_calling_agent (OpenAI function-calling under the hood)
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, Q1_TOOLS, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=Q1_TOOLS,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,   # Reliability #4: graceful error handling
    )
    return executor


def run_edtech_assistant(query: str, vector_store: FAISS, student_id: str = None):
    """
    End-to-End Flow:
    1. Receive user query
    2. Retrieve relevant context from vector store (RAG)
    3. Inject context into agent's system prompt
    4. Agent decides: answer from context OR call a tool
    5. Generate final grounded answer
    """
    print(f"\n{'='*60}")
    print(f"[Q1] User Query: {query}")
    print(f"{'='*60}")

    # Step 2: Retrieve context
    context = retrieve_context(vector_store, query)
    print(f"[RAG] Retrieved context ({len(context)} chars)")

    # Step 3+4+5: Agent runs with context injected
    agent_executor = create_edtech_agent(vector_store)

    # Append student_id hint if provided (helps agent call tools correctly)
    full_input = query
    if student_id:
        full_input += f" (My student ID is {student_id})"

    result = agent_executor.invoke({
        "input": full_input,
        "context": context,
    })

    print(f"\n[FINAL ANSWER]\n{result['output']}")
    return result["output"]

# =============================================================================
#  QUESTION 2 — MULTI-AGENT WORKFLOW WITH LANGGRAPH (Fintech Support)
# =============================================================================

from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
import operator

# ── 2A. STATE DEFINITION ──────────────────────────────────────────────────────

class SupportState(TypedDict):
    """
    Shared state that evolves across all agents in the graph.
    Each agent reads from and writes to this state.
    """
    query: str                          # Original user query (immutable)
    category: Optional[str]             # Classified category (set by Router)
    context: Optional[str]              # Retrieved docs/data (set by Retrieval Agent)
    reasoning: Optional[str]            # Step-by-step reasoning (set by Reasoning Agent)
    final_answer: Optional[str]         # Generated answer (set by Answer Agent)
    validation_passed: Optional[bool]   # Validation result (set by Validation Agent)
    retry_count: Annotated[int, operator.add]  # Auto-increments on retry
    error: Optional[str]                # Error message if something fails


# ── Mock fintech data ─────────────────────────────────────────────────────────
FINTECH_FAQ = {
    "upi limit": "UPI transaction limit is ₹1,00,000 per transaction and ₹2,00,000 per day.",
    "kyc": "KYC verification takes 24-48 hours. Upload Aadhaar + PAN on the app.",
    "interest rate": "Personal loan interest rates range from 10.5% to 18% per annum.",
    "account closure": "To close your account, visit the nearest branch with your ID proof.",
}

MOCK_TRANSACTIONS = {
    "TXN001": {"amount": 5000, "status": "success", "merchant": "Amazon", "date": "2026-03-20"},
    "TXN002": {"amount": 50000, "status": "flagged", "merchant": "Unknown Merchant", "date": "2026-03-22"},
    "TXN003": {"amount": 1200, "status": "failed", "merchant": "Swiggy", "date": "2026-03-23"},
}

# ── 2B. AGENT NODES ───────────────────────────────────────────────────────────

from langchain_core.messages import HumanMessage

def router_agent(state: SupportState) -> SupportState:
    """
    AGENT 1 — Router Agent
    Role: Classify the incoming query into a category.
    Input:  state["query"]
    Output: state["category"] ∈ {transaction, fraud, refund, faq, unknown}
    
    Uses LLM to classify so it handles natural language variations.
    """
    print(f"\n[Router Agent] Classifying query...")
    prompt = f"""Classify this customer support query into exactly one category.
Categories: transaction, fraud, refund, faq
Query: "{state['query']}"
Reply with only the category word."""

    response = llm.invoke([HumanMessage(content=prompt)])
    category = response.content.strip().lower()

    # Fallback if LLM returns unexpected value
    if category not in ["transaction", "fraud", "refund", "faq"]:
        category = "faq"

    print(f"[Router Agent] Category → '{category}'")
    return {**state, "category": category}


def retrieval_agent(state: SupportState) -> SupportState:
    """
    AGENT 2 — Retrieval Agent
    Role: Fetch relevant data based on category.
           - For FAQ: keyword search in FAQ store
           - For transaction/fraud: look up mock transaction DB
           - For refund: fetch transaction + apply refund rules
    Input:  state["query"], state["category"]
    Output: state["context"] — raw retrieved data as string
    """
    print(f"\n[Retrieval Agent] Fetching data for category='{state['category']}'...")
    query = state["query"].lower()
    category = state["category"]
    context = ""

    if category == "faq":
        # Simple keyword match (in production: vector search)
        matched = [v for k, v in FINTECH_FAQ.items() if k in query]
        context = matched[0] if matched else "No specific FAQ found. Provide general guidance."

    elif category in ["transaction", "fraud"]:
        # Extract transaction ID if present
        txn_id = next((w for w in query.upper().split() if w.startswith("TXN")), None)
        if txn_id and txn_id in MOCK_TRANSACTIONS:
            t = MOCK_TRANSACTIONS[txn_id]
            context = (
                f"Transaction {txn_id}: ₹{t['amount']} to {t['merchant']} "
                f"on {t['date']} — Status: {t['status'].upper()}"
            )
        else:
            context = "Transaction ID not found. Ask user to provide valid TXN ID."

    elif category == "refund":
        txn_id = next((w for w in query.upper().split() if w.startswith("TXN")), None)
        if txn_id and txn_id in MOCK_TRANSACTIONS:
            t = MOCK_TRANSACTIONS[txn_id]
            if t["status"] == "failed":
                context = f"TXN {txn_id} failed (₹{t['amount']}). Auto-refund eligible within 5-7 business days."
            elif t["status"] == "flagged":
                context = f"TXN {txn_id} is flagged for fraud review. Refund on hold pending investigation."
            else:
                context = f"TXN {txn_id} was successful. Refund requires merchant dispute process."
        else:
            context = "No transaction ID found. Cannot process refund without TXN ID."

    print(f"[Retrieval Agent] Context: {context[:100]}...")
    return {**state, "context": context}


def reasoning_agent(state: SupportState) -> SupportState:
    """
    AGENT 3 — Reasoning Agent
    Role: Perform step-by-step reasoning using retrieved context.
           Chain-of-thought prompting for accuracy.
    Input:  state["query"], state["context"], state["category"]
    Output: state["reasoning"] — structured reasoning trace
    """
    print(f"\n[Reasoning Agent] Generating reasoning...")
    prompt = f"""You are a fintech support reasoning engine.

Customer Query: {state['query']}
Category: {state['category']}
Retrieved Data: {state['context']}

Think step by step:
1. What is the customer asking?
2. What does the retrieved data tell us?
3. What is the correct answer based on policy?
4. Are there any caveats or follow-up actions needed?

Provide your reasoning clearly."""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"[Reasoning Agent] Reasoning complete.")
    return {**state, "reasoning": response.content}


def answer_agent(state: SupportState) -> SupportState:
    """
    AGENT 4 — Answer Agent
    Role: Generate a clean, customer-friendly final answer from the reasoning.
    Input:  state["reasoning"], state["query"]
    Output: state["final_answer"]
    """
    print(f"\n[Answer Agent] Generating final answer...")
    prompt = f"""Based on this reasoning, write a clear, concise, friendly customer support reply.
Do NOT add information not present in the reasoning.

Reasoning:
{state['reasoning']}

Customer's original question: {state['query']}

Write the final reply (2-4 sentences max):"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"[Answer Agent] Answer generated.")
    return {**state, "final_answer": response.content}


def validation_agent(state: SupportState) -> SupportState:
    """
    AGENT 5 — Validation Agent
    Role: Verify the final answer is accurate, grounded, and not hallucinated.
           Checks: answer addresses the query, no contradictions with context.
    Input:  state["query"], state["context"], state["final_answer"]
    Output: state["validation_passed"] (True/False)
    
    This is a key reliability mechanism — catches bad answers before delivery.
    """
    print(f"\n[Validation Agent] Validating answer...")
    prompt = f"""You are a quality checker for a fintech support system.

Original Query: {state['query']}
Retrieved Context: {state['context']}
Generated Answer: {state['final_answer']}

Check:
1. Does the answer directly address the query? (yes/no)
2. Is the answer consistent with the retrieved context? (yes/no)
3. Does the answer contain any made-up information not in context? (yes/no)

Reply with ONLY: PASS or FAIL
- PASS if answer is accurate and grounded
- FAIL if answer is wrong, hallucinated, or off-topic"""

    response = llm.invoke([HumanMessage(content=prompt)])
    passed = "PASS" in response.content.upper()
    print(f"[Validation Agent] Result → {'✅ PASS' if passed else '❌ FAIL'}")
    return {**state, "validation_passed": passed}

# ── 2C. GRAPH CONSTRUCTION ────────────────────────────────────────────────────

def should_retry_or_end(state: SupportState) -> str:
    """
    Conditional edge after Validation Agent.
    - If validation passed → END
    - If failed and retry_count < 2 → go back to reasoning_agent (retry loop)
    - If failed and retry_count >= 2 → END with fallback (avoid infinite loop)
    """
    if state["validation_passed"]:
        return "end"
    if state["retry_count"] < 2:
        print(f"[Graph] Validation failed. Retrying... (attempt {state['retry_count'] + 1})")
        return "retry"
    print("[Graph] Max retries reached. Ending with best available answer.")
    return "end"


def build_fintech_graph() -> StateGraph:
    """
    Graph Structure:
    
    START
      │
      ▼
    router_agent          ← classifies query
      │
      ▼
    retrieval_agent       ← fetches relevant data
      │
      ▼
    reasoning_agent  ◄────────────────────┐
      │                                   │ retry (max 2x)
      ▼                                   │
    answer_agent                          │
      │                                   │
      ▼                                   │
    validation_agent ──── FAIL ───────────┘
      │
      PASS
      │
      ▼
     END
    """
    graph = StateGraph(SupportState)

    # Add all agent nodes
    graph.add_node("router",     router_agent)
    graph.add_node("retrieval",  retrieval_agent)
    graph.add_node("reasoning",  reasoning_agent)
    graph.add_node("answer",     answer_agent)
    graph.add_node("validation", validation_agent)

    # Define edges (flow)
    graph.set_entry_point("router")
    graph.add_edge("router",    "retrieval")
    graph.add_edge("retrieval", "reasoning")
    graph.add_edge("reasoning", "answer")
    graph.add_edge("answer",    "validation")

    # Conditional edge: validation → retry or end
    graph.add_conditional_edges(
        "validation",
        should_retry_or_end,
        {
            "retry": "reasoning",   # loop back to reasoning with same context
            "end":   END,
        }
    )

    return graph.compile()


def run_fintech_support(query: str):
    """
    End-to-End fintech support flow using the LangGraph multi-agent system.
    """
    print(f"\n{'='*60}")
    print(f"[Q2] Customer Query: {query}")
    print(f"{'='*60}")

    app = build_fintech_graph()

    # Initial state
    initial_state: SupportState = {
        "query": query,
        "category": None,
        "context": None,
        "reasoning": None,
        "final_answer": None,
        "validation_passed": None,
        "retry_count": 0,
        "error": None,
    }

    final_state = app.invoke(initial_state)

    print(f"\n{'─'*60}")
    print(f"[FINAL ANSWER]\n{final_state['final_answer']}")
    print(f"[Validation Passed: {final_state['validation_passed']}]")
    print(f"[Retries: {final_state['retry_count']}]")
    print(f"{'─'*60}")
    return final_state

# =============================================================================
#  MAIN — Run both systems with sample queries
# =============================================================================

if __name__ == "__main__":

    print("\n" + "█"*60)
    print("  QUESTION 1: Ed-Tech RAG + Agent System")
    print("█"*60)

    # Build vector store from mock docs
    vector_store = build_rag_pipeline(MOCK_DOCS)

    # Test 1: General policy question → should use RAG context
    run_edtech_assistant(
        query="What is the refund policy?",
        vector_store=vector_store,
    )

    # Test 2: Student-specific question → should call get_student_status tool
    run_edtech_assistant(
        query="What courses am I enrolled in and what is my progress?",
        vector_store=vector_store,
        student_id="S001",
    )

    # Test 3: Deadline question → should call get_assignment_deadlines tool
    run_edtech_assistant(
        query="When are the deadlines for Python Basics?",
        vector_store=vector_store,
    )

    # Test 4: Refund eligibility → should call check_refund_eligibility tool
    run_edtech_assistant(
        query="Am I eligible for a refund on Data Science 101?",
        vector_store=vector_store,
        student_id="S001",
    )

    print("\n" + "█"*60)
    print("  QUESTION 2: Fintech Multi-Agent LangGraph System")
    print("█"*60)

    # Test 1: FAQ query
    run_fintech_support("What is the UPI transaction limit?")

    # Test 2: Transaction query
    run_fintech_support("Can you check the status of my transaction TXN002?")

    # Test 3: Refund request
    run_fintech_support("I want a refund for transaction TXN003 which failed.")

    # Test 4: Fraud flag
    run_fintech_support("My transaction TXN002 looks suspicious, is it flagged for fraud?")
