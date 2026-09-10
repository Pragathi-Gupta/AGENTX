import io
import json
import os
import re
from datetime import datetime
from urllib.parse import parse_qs, unquote, urljoin, urlparse

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
# Gemini (Interactions API) + Web Search + Website Reader
# Files + Google Sheets + MCP + Conversation History
# ============================================================

st.set_page_config(
    page_title="AGENTX",
    page_icon="🤖",
    layout="wide",
)

MODEL_OPTIONS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]

DEFAULT_MODEL = "gemini-3.6-flash"

MAX_TOOL_ROUNDS = 6


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

if "model" not in st.session_state:
    st.session_state.model = DEFAULT_MODEL


# ============================================================
# TOOL 1 — WEB SEARCH (DuckDuckGo HTML)
# ============================================================

def clean_ddg_href(href: str) -> str:
    """
    DuckDuckGo wraps result links in a redirect URL.
    Unwrap it to get the real destination URL.
    """

    if not href:
        return ""

    if href.startswith("//"):
        href = "https:" + href

    try:
        parsed = urlparse(href)

        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            query = parse_qs(parsed.query)

            if "uddg" in query:
                return unquote(query["uddg"][0])
    except Exception:
        pass

    return href


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

        for item in soup.select(".result")[: max_results * 2]:

            title_element = item.select_one(".result__a")
            snippet_element = item.select_one(".result__snippet")

            if not title_element:
                continue

            title = title_element.get_text(" ", strip=True)
            link = clean_ddg_href(title_element.get("href", ""))

            snippet = ""

            if snippet_element:
                snippet = snippet_element.get_text(" ", strip=True)

            if not link or not link.startswith("http"):
                continue

            results.append({
                "title": title,
                "url": link,
                "snippet": snippet,
            })

            if len(results) >= max_results:
                break

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

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unnecessary elements
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
            tag.decompose()

        title = ""

        if soup.title:
            title = soup.title.get_text(" ", strip=True)

        text = soup.get_text(" ", strip=True)

        # Collect useful links (deduplicated, http/https only)
        links = []
        seen_links = set()

        for a in soup.find_all("a", href=True):

            href = urljoin(url, a["href"]).split("#")[0]

            if not href.startswith("http"):
                continue

            if href in seen_links:
                continue

            text_link = a.get_text(" ", strip=True)

            if not text_link:
                continue

            seen_links.add(href)
            links.append({
                "text": text_link[:150],
                "url": href,
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
    info = st.session_state.get("uploaded_info")

    if not info:
        return {"error": "No file is currently uploaded."}

    return info


def extract_file(uploaded_file):

    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    # CSV
    if name.endswith(".csv"):

        try:
            df = pd.read_csv(io.BytesIO(raw))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")

        return {
            "type": "CSV",
            "name": uploaded_file.name,
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "preview": df.head(20).to_dict(orient="records"),
        }

    # Excel
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

    # PDF
    if name.endswith(".pdf"):

        if PdfReader is None:
            return {"error": "PDF package is not installed (pip install pypdf)."}

        reader = PdfReader(io.BytesIO(raw))

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
            return {"error": "DOCX package is not installed (pip install python-docx)."}

        document = Document(io.BytesIO(raw))

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        # Include table content too
        for table in document.tables:
            for row in table.rows:
                text += "\n" + " | ".join(
                    cell.text for cell in row.cells
                )

        return {
            "type": "DOCX",
            "name": uploaded_file.name,
            "text": text[:30000],
        }

    return {"error": "Unsupported file type."}


# ============================================================
# TOOL 4 — GOOGLE SHEETS
# ============================================================

def read_google_sheet(sheet_url: str):

    try:
        match = re.search(
            r"/spreadsheets/d/([a-zA-Z0-9-_]+)",
            sheet_url,
        )

        if not match:
            return {"error": "Invalid Google Sheets URL."}

        sheet_id = match.group(1)

        csv_url = (
            "https://docs.google.com/spreadsheets/d/"
            f"{sheet_id}/export?format=csv"
        )

        response = requests.get(csv_url, timeout=20)

        if response.status_code != 200:
            return {
                "error": (
                    "Could not access the Google Sheet. "
                    "Make sure it is shared as "
                    "'Anyone with the link'."
                )
            }

        df = pd.read_csv(io.StringIO(response.text))

        return {
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "preview": df.head(20).to_dict(orient="records"),
        }

    except Exception as e:
        return {"error": f"Google Sheets error: {str(e)}"}


# ============================================================
# INTERACTION HELPERS
# ============================================================

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


def get_steps(interaction):
    """
    The Interactions API exposes model-generated steps on
    `interaction.outputs`. Fall back to `.steps` for older SDKs.
    """

    outputs = getattr(interaction, "outputs", None)

    if outputs is None:
        outputs = getattr(interaction, "steps", None)

    return outputs or []


def extract_final_text(interaction) -> str:
    """
    Collect the text outputs of the final interaction.
    """

    parts = []

    for step in get_steps(interaction):

        if step_value(step, "type") == "text":
            text = step_value(step, "text", "")

            if text:
                parts.append(text)

    return "\n\n".join(parts).strip()


def to_json_safe(obj):
    """
    Make tool results JSON-serializable (numpy ints, datetimes, etc.).
    """

    try:
        return json.loads(
            json.dumps(obj, ensure_ascii=False, default=str)
        )
    except Exception:
        return {"repr": repr(obj)[:2000]}


# ============================================================
# EXECUTE FUNCTION CALLS
# ============================================================

def execute_function_calls(client, interaction, tools, system_instruction, model):
    """
    Run every tool the model requested, feed the results back to
    Gemini, and repeat until the model produces a final answer.
    Returns (final_interaction, tool_events) for UI logging.
    """

    tool_events = []
    current = interaction

    for _ in range(MAX_TOOL_ROUNDS):

        function_steps = [
            step
            for step in get_steps(current)
            if step_value(step, "type") == "function_call"
        ]

        if not function_steps:
            break

        results = []

        for step in function_steps:

            name = step_value(step, "name", "")
            arguments = step_value(step, "arguments", {}) or {}

            # `arguments` can arrive as a JSON string
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments else {}
                except Exception:
                    arguments = {}

            # -------------------------
            # WEB SEARCH
            # -------------------------
            if name == "web_search":
                result = web_search(
                    str(arguments.get("query", "")),
                    int(arguments.get("max_results", 6) or 6),
                )

            # -------------------------
            # WEBSITE READER
            # -------------------------
            elif name == "read_webpage":
                result = read_webpage(str(arguments.get("url", "")))

            # -------------------------
            # FILE
            # -------------------------
            elif name == "analyze_uploaded_file":
                result = analyze_uploaded_file()

            # -------------------------
            # GOOGLE SHEETS
            # -------------------------
            elif name == "read_google_sheet":
                result = read_google_sheet(str(arguments.get("sheet_url", "")))

            else:
                result = {"error": f"Unknown tool: {name}"}

            safe_result = to_json_safe(result)

            tool_events.append({
                "tool": name,
                "arguments": to_json_safe(arguments),
                "summary": json.dumps(
                    safe_result, ensure_ascii=False, default=str
                )[:600],
            })

            results.append({
                "type": "function_result",
                "name": name,
                "call_id": step_value(step, "id", "") or "",
                "result": safe_result,
            })

        # Send tool results back to Gemini.
        # NOTE: tools, system_instruction and generation settings are
        # interaction-scoped — they must be re-sent on every call.
        current = client.interactions.create(
            model=model,
            system_instruction=system_instruction,
            tools=tools,
            previous_interaction_id=current.id,
            input=results,
        )

    return current, tool_events


# ============================================================
# API KEY (secrets.toml OR environment variable)
# ============================================================

try:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    api_key = ""

api_key = api_key or os.environ.get("GEMINI_API_KEY", "")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🤖 AGENTX")
st.sidebar.caption("Autonomous Multi-Tool AI Agent")

if not api_key:
    st.sidebar.error("GEMINI_API_KEY is missing.")

st.session_state.model = st.sidebar.selectbox(
    "Gemini model",
    MODEL_OPTIONS,
    index=MODEL_OPTIONS.index(DEFAULT_MODEL),
)


# ============================================================
# FILE CONNECTOR
# ============================================================

st.sidebar.divider()
st.sidebar.subheader("📂 File Connector")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV, Excel, PDF or DOCX",
    type=["csv", "xlsx", "pdf", "docx"],
)

if uploaded_file:
    try:
        st.session_state.uploaded_info = extract_file(uploaded_file)
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
    except Exception as e:
        st.session_state.uploaded_info = {"error": str(e)}
        st.sidebar.error(f"File error: {e}")


# ============================================================
# WEB CONNECTORS
# ============================================================

st.sidebar.subheader("🌐 Web Connectors")

enable_web = st.sidebar.checkbox("Enable Web Search", value=True)
enable_web_reader = st.sidebar.checkbox("Enable Website Reader", value=True)


# ============================================================
# MCP CONNECTOR
# ============================================================

st.sidebar.subheader("🔌 MCP Connector")

enable_mcp = st.sidebar.checkbox("Enable custom MCP server", value=False)

mcp_url = ""
mcp_name = "custom_mcp"
mcp_token = ""

if enable_mcp:
    mcp_name = st.sidebar.text_input("MCP server name", value="custom_mcp")
    mcp_url = st.sidebar.text_input(
        "MCP server URL",
        placeholder="https://your-server.com/mcp",
    )
    mcp_token = st.sidebar.text_input("Bearer token (optional)", type="password")


# ============================================================
# CONVERSATION HISTORY
# ============================================================

st.sidebar.subheader("💬 Conversation History")

if st.sidebar.button("🆕 New Conversation", use_container_width=True):

    if st.session_state.messages:
        st.session_state.chat_history.append({
            "title": st.session_state.messages[0]["content"][:60],
            "messages": [dict(m) for m in st.session_state.messages],
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "interaction_id": st.session_state.interaction_id,
        })

    st.session_state.messages = []
    st.session_state.interaction_id = None
    st.rerun()

if st.session_state.chat_history:
    st.sidebar.caption(
        f"{len(st.session_state.chat_history)} saved conversation(s)"
    )

for idx, saved in enumerate(st.session_state.chat_history):

    col_load, col_del = st.sidebar.columns([4, 1])

    if col_load.button(
        f"💬 {saved['title'][:36]}",
        key=f"load_chat_{idx}",
        use_container_width=True,
        help=f"Saved {saved.get('saved_at', '')}",
    ):
        st.session_state.messages = [
            dict(m) for m in saved["messages"]
        ]
        st.session_state.interaction_id = saved.get("interaction_id")
        st.rerun()

    if col_del.button("🗑", key=f"del_chat_{idx}"):
        st.session_state.chat_history.pop(idx)
        st.rerun()


# ============================================================
# MAIN PAGE
# ============================================================

st.title("🤖 AGENTX")
st.subheader("Autonomous Multi-Tool AI Agent")
st.write(
    "Search the web, read websites, analyze files, "
    "access Google Sheets and connect external tools."
)


# ---- Render the full conversation so far ---------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

system_instruction = """
You are AGENTX, an autonomous multi-tool AI agent.

You can:
1. Have normal conversations.
2. Search the public web.
3. Read public webpages.
4. Analyze uploaded CSV, Excel, PDF and DOCX files.
5. Read public Google Sheets.
6. Use configured MCP tools.
7. Perform multiple tool calls in sequence.

IMPORTANT:
When current or online information is needed, use the web_search tool.
When detailed information from a webpage is needed, use read_webpage.
When the user asks about an uploaded file, use analyze_uploaded_file.
When the user gives a Google Sheets URL, use read_google_sheet.

Use tools whenever they are genuinely useful.

Never claim that you searched the web, opened a website, analyzed a file,
or used MCP unless the tool actually ran.

When web search is used, include the useful source URLs in your answer.

Be clear, concise and helpful.

Act like an autonomous agent rather than a simple chatbot.
"""


# ============================================================
# STOP EARLY IF NO API KEY (before the chat input)
# ============================================================

if not api_key:
    st.info(
        "Add GEMINI_API_KEY in Streamlit → Manage app → Secrets, "
        "or set it as an environment variable, then restart the app."
    )
    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=api_key)


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
            "Search the public internet for current or factual information. "
            "Use this when the user asks for latest information, news, "
            "websites, research, current facts or online resources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results.",
                },
            },
            "required": ["query"],
        },
    })


# WEBSITE READER
if enable_web_reader:
    tools.append({
        "type": "function",
        "name": "read_webpage",
        "description": (
            "Open and read a public webpage. Use this after web search "
            "when the user needs details from a website."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The complete webpage URL.",
                },
            },
            "required": ["url"],
        },
    })


# FILE TOOL
if st.session_state.uploaded_info:
    tools.append({
        "type": "function",
        "name": "analyze_uploaded_file",
        "description": (
            "Analyze the currently uploaded CSV, Excel, PDF or DOCX file."
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
        "Read and analyze a public Google Sheet when the user provides "
        "a Google Sheets URL."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sheet_url": {
                "type": "string",
                "description": "Complete Google Sheets URL.",
            },
        },
        "required": ["sheet_url"],
    },
})


# MCP
if enable_mcp and mcp_url:
    mcp_headers = {}

    if mcp_token:
        mcp_headers = {"Authorization": f"Bearer {mcp_token}"}

    tools.append({
        "type": "mcp_server",
        "name": mcp_name,
        "url": mcp_url,
        "headers": mcp_headers,
    })


# ============================================================
# CHAT INPUT  (must be top-level, never inside sidebar/columns)
# ============================================================

prompt = st.chat_input(
    "Ask AGENTX anything — search the web, read a site, analyze a file…"
)


if prompt:

    # 1. Show and store the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run the agent turn
    with st.chat_message("assistant"):

        assistant_text = ""
        tool_events = []

        with st.spinner("AGENTX is working…"):

            try:
                create_kwargs = {
                    "model": st.session_state.model,
                    "system_instruction": system_instruction,
                    "input": prompt,
                }

                if tools:
                    create_kwargs["tools"] = tools

                if st.session_state.interaction_id:
                    create_kwargs["previous_interaction_id"] = (
                        st.session_state.interaction_id
                    )

                interaction = client.interactions.create(**create_kwargs)

                # Run requested tools and feed results back to Gemini
                interaction, tool_events = execute_function_calls(
                    client,
                    interaction,
                    tools,
                    system_instruction,
                    st.session_state.model,
                )

                st.session_state.interaction_id = getattr(
                    interaction, "id", None
                )

                assistant_text = extract_final_text(interaction)

            except Exception as e:
                assistant_text = f"⚠️ **AGENTX hit an error:** {e}"

        if not assistant_text:
            assistant_text = "⚠️ The model returned no text output."

        st.markdown(assistant_text)

        # Show which tools ran during this turn
        if tool_events:
            with st.expander(f"🔧 Tools used ({len(tool_events)})"):
                for event in tool_events:
                    args_json = json.dumps(
                        event["arguments"], ensure_ascii=False
                    )
                    st.markdown(f"**`{event['tool']}`**  `{args_json}`")
                    st.code(event["summary"], language="json")

    # 3. Store the assistant reply
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_text,
    })
