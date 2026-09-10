import io
import json
import re
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
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
# AGENTX
# Gemini 3.6 Flash + Web Search + Website Reader
# Files + Google Sheets + MCP + Conversation History
# ============================================================

st.set_page_config(
    page_title="AGENTX",
    page_icon="🤖",
    layout="wide",
)

MODEL = "gemini-3.6-flash"


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "interaction_id" not in st.session_state:
    st.session_state.interaction_id = None

if "uploaded_info" not in st.session_state:
    st.session_state.uploaded_info = None


# ============================================================
# TOOL 1 — WEB SEARCH
# ============================================================

def web_search(query: str, max_results: int = 6):
    """
    Search the public web using DuckDuckGo HTML.
    This is a custom tool and does not use Gemini Search Grounding.
    """

    try:
        url = "https://html.duckduckgo.com/html/"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/131 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            params={"q": query},
            headers=headers,
            timeout=20,
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        results = []

        for item in soup.select(".result")[:max_results]:

            title_element = item.select_one(".result__a")
            snippet_element = item.select_one(".result__snippet")

            if not title_element:
                continue

            title = title_element.get_text(
                " ",
                strip=True
            )

            link = title_element.get("href", "")

            snippet = ""

            if snippet_element:
                snippet = snippet_element.get_text(
                    " ",
                    strip=True
                )

            results.append({
                "title": title,
                "url": link,
                "snippet": snippet,
            })

        if not results:
            return {
                "error": "No search results were found.",
                "query": query,
            }

        return {
            "query": query,
            "results": results,
        }

    except Exception as e:

        return {
            "error": f"Web search failed: {str(e)}"
        }


# ============================================================
# TOOL 2 — READ WEBSITE
# ============================================================

def read_webpage(url: str):
    """
    Fetch and extract readable text from a public webpage.
    """

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/131 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=20,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary elements
        for tag in soup([
            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "footer"
        ]):
            tag.decompose()

        title = ""

        if soup.title:
            title = soup.title.get_text(
                " ",
                strip=True
            )

        text = soup.get_text(
            " ",
            strip=True
        )

        # Collect useful links
        links = []

        for a in soup.find_all("a", href=True):

            href = urljoin(
                url,
                a["href"]
            )

            text_link = a.get_text(
                " ",
                strip=True
            )

            if text_link:
                links.append({
                    "text": text_link[:150],
                    "url": href
                })

            if len(links) >= 30:
                break

        return {
            "url": url,
            "title": title,
            "text": text[:30000],
            "links": links,
        }

    except Exception as e:

        return {
            "error": f"Website reading failed: {str(e)}",
            "url": url,
        }


# ============================================================
# TOOL 3 — FILE ANALYZER
# ============================================================

def analyze_uploaded_file():

    info = st.session_state.get(
        "uploaded_info"
    )

    if not info:
        return {
            "error": "No file is currently uploaded."
        }

    return info


def extract_file(uploaded_file):

    name = uploaded_file.name.lower()

    raw = uploaded_file.getvalue()


    # CSV
    if name.endswith(".csv"):

        df = pd.read_csv(
            io.BytesIO(raw)
        )

        return {
            "type": "CSV",
            "name": uploaded_file.name,
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": (
                df.isnull()
                .sum()
                .to_dict()
            ),
            "preview": df.head(20).to_dict(
                orient="records"
            ),
        }


    # Excel
    if name.endswith(".xlsx"):

        df = pd.read_excel(
            io.BytesIO(raw)
        )

        return {
            "type": "Excel",
            "name": uploaded_file.name,
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": (
                df.isnull()
                .sum()
                .to_dict()
            ),
            "preview": df.head(20).to_dict(
                orient="records"
            ),
        }


    # PDF
    if name.endswith(".pdf"):

        if PdfReader is None:

            return {
                "error": "PDF package is not installed."
            }

        reader = PdfReader(
            io.BytesIO(raw)
        )

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        return {
            "type": "PDF",
            "name": uploaded_file.name,
            "pages": len(reader.pages),
            "text": text[:30000],
        }


    # DOCX
    if name.endswith(".docx"):

        if Document is None:

            return {
                "error": "DOCX package is not installed."
            }

        document = Document(
            io.BytesIO(raw)
        )

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        return {
            "type": "DOCX",
            "name": uploaded_file.name,
            "text": text[:30000],
        }


    return {
        "error": "Unsupported file type."
    }


# ============================================================
# TOOL 4 — GOOGLE SHEETS
# ============================================================

def read_google_sheet(sheet_url: str):

    try:

        match = re.search(
            r"/spreadsheets/d/([a-zA-Z0-9-_]+)",
            sheet_url
        )

        if not match:

            return {
                "error": "Invalid Google Sheets URL."
            }

        sheet_id = match.group(1)

        csv_url = (
            "https://docs.google.com/spreadsheets/d/"
            f"{sheet_id}/export?format=csv"
        )

        response = requests.get(
            csv_url,
            timeout=20
        )

        if response.status_code != 200:

            return {
                "error": (
                    "Could not access the Google Sheet. "
                    "Make sure it is shared as "
                    "'Anyone with the link'."
                )
            }

        df = pd.read_csv(
            io.StringIO(
                response.text
            )
        )

        return {
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": (
                df.isnull()
                .sum()
                .to_dict()
            ),
            "preview": df.head(20).to_dict(
                orient="records"
            ),
        }

    except Exception as e:

        return {
            "error": f"Google Sheets error: {str(e)}"
        }


# ============================================================
# HELPER
# ============================================================

def step_value(step, key, default=None):

    try:

        value = getattr(
            step,
            key
        )

        if value is not None:
            return value

    except Exception:
        pass

    try:

        if hasattr(
            step,
            "model_dump"
        ):

            return step.model_dump().get(
                key,
                default
            )

    except Exception:
        pass

    return default


# ============================================================
# EXECUTE FUNCTION CALLS
# ============================================================

def execute_function_calls(
    client,
    interaction,
    tools
):

    current = interaction

    for _ in range(6):

        function_steps = [
            step
            for step in getattr(
                current,
                "steps",
                []
            )
            if step_value(
                step,
                "type"
            ) == "function_call"
        ]

        if not function_steps:

            return current

        results = []

        for step in function_steps:

            name = step_value(
                step,
                "name",
                ""
            )

            arguments = step_value(
                step,
                "arguments",
                {}
            ) or {}


            # -------------------------
            # WEB SEARCH
            # -------------------------

            if name == "web_search":

                result = web_search(
                    arguments.get(
                        "query",
                        ""
                    ),
                    int(
                        arguments.get(
                            "max_results",
                            6
                        )
                    )
                )


            # -------------------------
            # WEBSITE READER
            # -------------------------

            elif name == "read_webpage":

                result = read_webpage(
                    arguments.get(
                        "url",
                        ""
                    )
                )


            # -------------------------
            # FILE
            # -------------------------

            elif name == "analyze_uploaded_file":

                result = analyze_uploaded_file()


            # -------------------------
            # GOOGLE SHEETS
            # -------------------------

            elif name == "read_google_sheet":

                result = read_google_sheet(
                    arguments.get(
                        "sheet_url",
                        ""
                    )
                )


            else:

                result = {
                    "error": (
                        f"Unknown tool: {name}"
                    )
                }


            results.append({

                "type": "function_result",

                "name": name,

                "call_id": step_value(
                    step,
                    "id",
                    ""
                ),

                "result": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            result,
                            ensure_ascii=False,
                            default=str
                        )
                    }
                ],

            })


        # Send tool results back to Gemini

        current = client.interactions.create(

            model=MODEL,

            previous_interaction_id=current.id,

            tools=tools,

            input=results,

        )

    return current


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🤖 AGENTX")

st.sidebar.caption(
    "Autonomous Multi-Tool AI Agent"
)


api_key = st.secrets.get(
    "GEMINI_API_KEY",
    ""
)


if not api_key:

    st.sidebar.error(
        "GEMINI_API_KEY is missing."
    )


# ============================================================
# FILE CONNECTOR
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "📂 File Connector"
)

uploaded_file = st.sidebar.file_uploader(

    "Upload CSV, Excel, PDF or DOCX",

    type=[
        "csv",
        "xlsx",
        "pdf",
        "docx"
    ],

)


if uploaded_file:

    try:

        st.session_state.uploaded_info = (
            extract_file(
                uploaded_file
            )
        )

        st.sidebar.success(
            f"Loaded: {uploaded_file.name}"
        )

    except Exception as e:

        st.session_state.uploaded_info = {
            "error": str(e)
        }

        st.sidebar.error(
            f"File error: {e}"
        )


# ============================================================
# WEB CONNECTORS
# ============================================================

st.sidebar.subheader(
    "🌐 Web Connectors"
)

enable_web = st.sidebar.checkbox(
    "Enable Web Search",
    value=True
)

enable_web_reader = st.sidebar.checkbox(
    "Enable Website Reader",
    value=True
)


# ============================================================
# MCP CONNECTOR
# ============================================================

st.sidebar.subheader(
    "🔌 MCP Connector"
)

enable_mcp = st.sidebar.checkbox(
    "Enable custom MCP server",
    value=False
)

mcp_url = ""
mcp_name = "custom_mcp"
mcp_token = ""

if enable_mcp:

    mcp_name = st.sidebar.text_input(
        "MCP server name",
        value="custom_mcp"
    )

    mcp_url = st.sidebar.text_input(
        "MCP server URL",
        placeholder="https://your-server.com/mcp"
    )

    mcp_token = st.sidebar.text_input(
        "Bearer token (optional)",
        type="password"
    )


# ============================================================
# HISTORY
# ============================================================

st.sidebar.subheader(
    "💬 Conversation History"
)


if st.sidebar.button(
    "🆕 New Conversation",
    use_container_width=True
):

    if st.session_state.messages:

        st.session_state.chat_history.append({

            "title":
                st.session_state.messages[0][
                    "content"
                ][:60],

            "messages":
                st.session_state.messages.copy(),

            "saved_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                ),

        })


    st.session_state.messages = []

    st.session_state.interaction_id = None

    st.rerun()


if st.session_state.chat_history:

    st.sidebar.caption(
        f"{len(st.session_state.chat_history)} "
        "saved conversation(s)"
    )


# ============================================================
# MAIN PAGE
# ============================================================

st.title(
    "🤖 AGENTX"
)

st.subheader(
    "Autonomous Multi-Tool AI Agent"
)

st.write(
    "Search the web, read websites, analyze files, "
    "access Google Sheets and connect external tools."
)


if not api_key:

    st.info(
        "Add GEMINI_API_KEY in "
        "Streamlit → Manage app → Secrets."
    )

    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=api_key
)


# ============================================================
# TOOL DEFINITIONS
# ============================================================

tools = []


# WEB SEARCH

if enable_web:

    tools.append({

        "type": "function",

        "name": "web_search",

        "description": (
            "Search the public internet for "
            "current or factual information. "
            "Use this when the user asks for "
            "latest information, news, websites, "
            "research, current facts or online resources."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "query": {

                    "type": "string",

                    "description":
                        "The web search query."

                },

                "max_results": {

                    "type": "integer",

                    "description":
                        "Maximum number of search results."

                },

            },

            "required": [
                "query"
            ],

        },

    })


# WEBSITE READER

if enable_web_reader:

    tools.append({

        "type": "function",

        "name": "read_webpage",

        "description": (
            "Open and read a public webpage. "
            "Use this after web search when "
            "the user needs details from a website."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "url": {

                    "type": "string",

                    "description":
                        "The complete webpage URL."

                },

            },

            "required": [
                "url"
            ],

        },

    })


# FILE TOOL

if st.session_state.uploaded_info:

    tools.append({

        "type": "function",

        "name": "analyze_uploaded_file",

        "description": (
            "Analyze the currently uploaded "
            "CSV, Excel, PDF or DOCX file."
        ),

        "parameters": {

            "type": "object",

            "properties": {},

        },

    })


# GOOGLE SHEETS

tools.append({

    "type": "function",

    "name": "read_google_sheet",

    "description": (
        "Read and analyze a public Google Sheet "
        "when the user provides a Google Sheets URL."
    ),

    "parameters": {

        "type": "object",

        "properties": {

            "sheet_url": {

                "type": "string",

                "description":
                    "Complete Google Sheets URL."

            },

        },

        "required": [
            "sheet_url"
        ],

    },

})


# MCP

if enable_mcp and mcp_url:

    mcp_headers = {}

    if mcp_token:

        mcp_headers = {
            "Authorization":
                f"Bearer {mcp_token}"
        }

    tools.append({

        "type": "mcp_server",

        "name": mcp_name,

        "url": mcp_url,

        "headers": mcp_headers,

    })


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

system_instruction = """

You are AGENTX, an autonomous multi-tool AI agent.

You can:

1. Have normal
