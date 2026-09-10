import io
import json
import re
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from google import genai

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None


# ============================================================
# AGENTX — Gemini multi-tool agent
# Web search + URL reading + files + Google Sheets + MCP
# ============================================================

st.set_page_config(
    page_title="AGENTX",
    page_icon="🤖",
    layout="wide",
)

MODEL = "gemini-3.6-flash"

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "interaction_id" not in st.session_state:
    st.session_state.interaction_id = None

if "uploaded_info" not in st.session_state:
    st.session_state.uploaded_info = None


# -----------------------------
# Helpers
# -----------------------------
def extract_text_from_file(uploaded_file):
    """Return a compact text representation of an uploaded file."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
        return {
            "type": "CSV",
            "name": uploaded_file.name,
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "preview": df.head(20).to_dict(orient="records"),
        }

    if name.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(raw))
        return {
            "type": "Excel",
            "name": uploaded_file.name,
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "preview": df.head(20).to_dict(orient="records"),
        }

    if name.endswith(".pdf"):
        if PdfReader is None:
            return {"error": "PDF support package is not installed."}
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        return {
            "type": "PDF",
            "name": uploaded_file.name,
            "pages": len(reader.pages),
            "text": text[:30000],
        }

    if name.endswith(".docx"):
        if Document is None:
            return {"error": "DOCX support package is not installed."}
        doc = Document(io.BytesIO(raw))
        text = "\n".join(p.text for p in doc.paragraphs)
        return {
            "type": "DOCX",
            "name": uploaded_file.name,
            "text": text[:30000],
        }

    return {"error": "Unsupported file type."}


def read_google_sheet(sheet_url: str):
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not match:
        return {"error": "Invalid Google Sheets URL."}

    sheet_id = match.group(1)
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/"
        f"{sheet_id}/export?format=csv"
    )

    response = requests.get(csv_url, timeout=20)
    if response.status_code != 200:
        return {
            "error": (
                "Could not access the Google Sheet. "
                "Make sure it is shared as 'Anyone with the link'."
            )
        }

    df = pd.read_csv(io.StringIO(response.text))

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "preview": df.head(20).to_dict(orient="records"),
    }


def get_output_text(interaction):
    """Safely extract model output text from an Interaction."""
    try:
        if getattr(interaction, "output_text", None):
            return interaction.output_text
    except Exception:
        pass

    pieces = []
    try:
        for step in interaction.steps:
            if getattr(step, "type", "") == "model_output":
                for block in getattr(step, "content", []) or []:
                    if getattr(block, "type", "") == "text":
                        pieces.append(getattr(block, "text", ""))
    except Exception:
        pass

    return "\n".join(pieces).strip() or "I completed the tool step but received no text output."


def step_value(step, key, default=None):
    try:
        value = getattr(step, key)
        if value is not None:
            return value
    except Exception:
        pass
    try:
        if hasattr(step, "model_dump"):
            return step.model_dump().get(key, default)
    except Exception:
        pass
    return default


def run_function_calls(client, interaction, tools):
    """ Execute AGENTX's local function tools and continue the Interaction. Returns the final interaction. """
    current = interaction

    for _ in range(5):
        function_steps = [
            s for s in getattr(current, "steps", [])
            if step_value(s, "type") == "function_call"
        ]

        if not function_steps:
            return current

        results = []

        for step in function_steps:
            name = step_value(step, "name", "")
            args = step_value(step, "arguments", {}) or {}

            if name == "analyze_uploaded_file":
                result = st.session_state.get("uploaded_info")
                if not result:
                    result = {"error": "No file is currently uploaded."}

            elif name == "read_google_sheet":
                result = read_google_sheet(args.get("sheet_url", ""))

            else:
                result = {"error": f"Unknown local tool: {name}"}

            results.append({
                "type": "function_result",
                "name": name,
                "call_id": step_value(step, "id", ""),
                "result": [
                    {"type": "text", "text": json.dumps(result, default=str)}
                ],
            })

        current = client.interactions.create(
            model=MODEL,
            previous_interaction_id=current.id,
            tools=tools,
            input=results,
        )

    return current


# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("🤖 AGENTX")
st.sidebar.caption("Multi-tool AI agent")

api_key = st.secrets.get("GEMINI_API_KEY", "")

if not api_key:
    st.sidebar.error("GEMINI_API_KEY is missing in Streamlit Secrets.")

st.sidebar.divider()

# File connector
st.sidebar.subheader("📂 File Connector")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV, Excel, PDF or DOCX",
    type=["csv", "xlsx", "pdf", "docx"],
)

if uploaded_file is not None:
    try:
        st.session_state.uploaded_info = extract_text_from_file(uploaded_file)
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
    except Exception as e:
        st.session_state.uploaded_info = {"error": str(e)}
        st.sidebar.error(f"File error: {e}")

# Web connectors
st.sidebar.subheader("🌐 Web Connectors")
enable_web_search = st.sidebar.checkbox("Google Web Search", value=True)
enable_url_context = st.sidebar.checkbox("Website / URL Reader", value=True)

# MCP connector
st.sidebar.subheader("🔌 MCP Connector")
enable_mcp = st.sidebar.checkbox("Enable custom MCP server", value=False)

mcp_name = ""
mcp_url = ""
mcp_headers = {}

if enable_mcp:
    mcp_name = st.sidebar.text_input(
        "MCP server name",
        value="custom_mcp",
        help="Use letters, numbers and underscores. Avoid hyphens.",
    ).strip()

    mcp_url = st.sidebar.text_input(
        "MCP server URL",
        placeholder="https://your-server.example.com/mcp",
    ).strip()

    mcp_token = st.sidebar.text_input(
        "Bearer token (optional)",
        type="password",
    ).strip()

    if mcp_token:
        mcp_headers = {"Authorization": f"Bearer {mcp_token}"}

    if mcp_url:
        st.sidebar.info("AGENTX will expose the configured MCP server to Gemini.")

# History controls
st.sidebar.subheader("💬 History")

if st.sidebar.button("🆕 New conversation", use_container_width=True):
    if st.session_state.messages:
        st.session_state.chat_history.append({
            "title": st.session_state.messages[0]["content"][:60],
            "messages": st.session_state.messages.copy(),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

    st.session_state.messages = []
    st.session_state.interaction_id = None
    st.rerun()

if st.session_state.chat_history:
    st.sidebar.caption(f"{len(st.session_state.chat_history)} saved session(s)")
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        label = chat["title"] or f"Conversation {i + 1}"
        if st.sidebar.button(label, key=f"history_{i}", use_container_width=True):
            st.session_state.messages = chat["messages"].copy()
            st.session_state.interaction_id = None
            st.rerun()


# ============================================================
# Main UI
# ============================================================
st.title("🤖 AGENTX")
st.subheader("Your multi-tool AI agent")

st.markdown(
    "Search the web, read websites, analyze files, access Google Sheets, "
    "and connect external MCP tools."
)

if not api_key:
    st.info("Add GEMINI_API_KEY under Streamlit → Manage app → Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)


# ============================================================
# Tool declarations
# ============================================================
tools = []

if enable_web_search:
    tools.append({"type": "google_search"})

if enable_url_context:
    tools.append({"type": "url_context"})

if st.session_state.uploaded_info:
    tools.append({
        "type": "function",
        "name": "analyze_uploaded_file",
        "description": (
            "Analyze the currently uploaded CSV, Excel, PDF or DOCX file. "
            "Use this when the user asks about the uploaded file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What the user wants to know about the uploaded file.",
                }
            },
            "required": ["question"],
        },
    })

tools.append({
    "type": "function",
    "name": "read_google_sheet",
    "description": (
        "Read a public Google Sheet and analyze its tabular data. "
        "Use this when the user provides a Google Sheets URL."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sheet_url": {
                "type": "string",
                "description": "The complete public Google Sheets URL.",
            }
        },
        "required": ["sheet_url"],
    },
})

if enable_mcp and mcp_url:
    tools.append({
        "type": "mcp_server",
        "name": mcp_name or "custom_mcp",
        "url": mcp_url,
        "headers": mcp_headers,
    })


system_instruction = """ You are AGENTX, an autonomous multi-tool AI agent. Your responsibilities: - Answer normal questions conversationally. - Use Google Search for current or factual information that benefits from fresh web data. - Use URL Context when you need to read or deeply analyze a webpage. - Use the uploaded-file tool when the user asks about an uploaded file. - Use the Google Sheets tool when the user provides a Google Sheets URL. - Use configured MCP tools when they are relevant to the user's request. - Never claim that you searched, scraped, opened, or used a connector unless the tool actually ran. - When web tools are used, provide useful source links/citations when available. - Explain which tools you used when that helps the user understand the result. - Prefer performing actions over merely explaining how the user could do them. """


# ============================================================
# Render previous messages
# ============================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# Chat
# ============================================================
user_prompt = st.chat_input("Ask AGENTX anything...")

if user_prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt,
    })

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        try:
            # Stateful server-side conversation when possible.
            if st.session_state.interaction_id:
                interaction = client.interactions.create(
                    model=MODEL,
                    previous_interaction_id=st.session_state.interaction_id,
                    input=user_prompt,
                    tools=tools,
                    system_instruction=system_instruction,
                )
            else:
                interaction = client.interactions.create(
                    model=MODEL,
                    input=user_prompt,
                    tools=tools,
                    system_instruction=system_instruction,
                )

            # Execute local function calls, if Gemini requests them.
            interaction = run_function_calls(client, interaction, tools)

            answer = get_output_text(interaction)

            st.session_state.interaction_id = interaction.id

            st.markdown(answer)

            # Show a compact tool trace for the demo.
            used_tools = []
            for step in getattr(interaction, "steps", []):
                step_type = step_value(step, "type", "")
                if step_type in {
                    "google_search_call",
                    "url_context_call",
                    "function_call",
                    "mcp_server_tool_call",
                }:
                    used_tools.append(step_type)

            if used_tools:
                st.caption(
                    "🛠️ Tools used: " +
                    ", ".join(dict.fromkeys(used_tools))
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception as e:
            error_message = (
                f"❌ **Agent error**\n\n"
                f"`{str(e)}`\n\n"
                "If the error is related to MCP, verify that the server is "
                "a Streamable HTTP MCP endpoint and that its authentication "
                "headers are correct."
            )
            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
            })
