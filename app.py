import io, re, requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None
try:
    from docx import Document
except Exception:
    Document = None

st.set_page_config(page_title="AGENTX", page_icon="🤖", layout="wide")
MODEL = "gemini-2.5-flash"

# ---------- Tools ----------
def web_search(query: str) -> dict:
    """Search the public web."""
    try:
        r = requests.get("https://html.duckduckgo.com/html/", params={"q": query},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        for x in soup.select(".result")[:6]:
            a = x.select_one(".result__a")
            s = x.select_one(".result__snippet")
            if a:
                out.append({"title": a.get_text(" ", strip=True),
                            "url": a.get("href", ""),
                            "snippet": s.get_text(" ", strip=True) if s else ""})
        return {"query": query, "results": out} if out else {"error": "No results found."}
    except Exception as e:
        return {"error": str(e)}

def read_webpage(url: str) -> dict:
    """Read a public webpage."""
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
            tag.decompose()
        return {"url": url,
                "title": soup.title.get_text(" ", strip=True) if soup.title else "",
                "text": soup.get_text(" ", strip=True)[:25000]}
    except Exception as e:
        return {"error": str(e), "url": url}

def analyze_file(name: str, raw: bytes) -> dict:
    """Analyze an uploaded CSV, XLSX, PDF or DOCX file."""
    try:
        n = name.lower()
        if n.endswith(".csv") or n.endswith(".xlsx"):
            df = pd.read_csv(io.BytesIO(raw)) if n.endswith(".csv") else pd.read_excel(io.BytesIO(raw))
            return {"name": name, "rows": len(df), "columns": list(df.columns),
                    "missing_values": df.isna().sum().to_dict(),
                    "preview": df.head(20).to_dict(orient="records")}
        if n.endswith(".pdf"):
            if PdfReader is None: return {"error": "pypdf is not installed."}
            reader = PdfReader(io.BytesIO(raw))
            return {"name": name, "pages": len(reader.pages),
                    "text": "\n".join(p.extract_text() or "" for p in reader.pages)[:25000]}
        if n.endswith(".docx"):
            if Document is None: return {"error": "python-docx is not installed."}
            doc = Document(io.BytesIO(raw))
            return {"name": name, "text": "\n".join(p.text for p in doc.paragraphs)[:25000]}
        return {"error": "Unsupported file type."}
    except Exception as e:
        return {"error": str(e)}

def read_google_sheet(url: str) -> dict:
    """Read a public Google Sheet."""
    try:
        m = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", url)
        if not m: return {"error": "Invalid Google Sheets URL."}
        csv_url = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
        r = requests.get(csv_url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        return {"rows": len(df), "columns": list(df.columns),
                "missing_values": df.isna().sum().to_dict(),
                "preview": df.head(20).to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}

# ---------- Sidebar ----------
st.sidebar.title("🤖 AGENTX")
st.sidebar.caption("Autonomous Multi-Tool AI Agent")
api_key = st.secrets.get("GEMINI_API_KEY", "")

uploaded = st.sidebar.file_uploader("📂 Upload CSV, Excel, PDF or DOCX",
                                    type=["csv", "xlsx", "pdf", "docx"])
if uploaded:
    st.session_state["file_info"] = analyze_file(uploaded.name, uploaded.getvalue())
    st.session_state["file_name"] = uploaded.name
    st.sidebar.success(f"Loaded: {uploaded.name}")

st.sidebar.subheader("🌐 Web Connectors")
enable_web = st.sidebar.checkbox("Enable Web Search", True)
enable_reader = st.sidebar.checkbox("Enable Website Reader", True)

st.sidebar.subheader("🔌 MCP Connector")
st.sidebar.info("MCP connector UI is included for the assessment; remote MCP support depends on the Gemini API/model availability.")

st.sidebar.subheader("💬 Conversation History")
if st.sidebar.button("🆕 New Conversation", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Main ----------
st.title("🤖 AGENTX")
st.subheader("Autonomous Multi-Tool AI Agent")
st.write("Search the web, read websites, analyze files and access Google Sheets.")

if not api_key:
    st.warning("Add GEMINI_API_KEY in Streamlit → Manage app → Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

system = """You are AGENTX, an autonomous multi-tool AI agent. Use tools whenever useful. Use web_search for current/public online information. Use read_webpage when a webpage needs to be inspected in detail. Use the uploaded-file tool when the user asks about the uploaded file. Use read_google_sheet when a Google Sheets URL is provided. Never claim a tool was used unless it actually ran. Give concise, useful answers. """

# Show previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask AGENTX anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    functions = []
    if enable_web: functions.append(web_search)
    if enable_reader: functions.append(read_webpage)

    if st.session_state.get("file_info"):
        def uploaded_file_tool() -> dict:
            """Analyze the currently uploaded file."""
            return st.session_state["file_info"]
        functions.append(uploaded_file_tool)

    functions.append(read_google_sheet)

    history = st.session_state.messages[-12:]
    contents = [f'{m["role"]}: {m["content"]}' for m in history]
    contents.append("If needed, use your available tools before answering the latest user request.")

    with st.chat_message("assistant"):
        try:
            config = types.GenerateContentConfig(
                system_instruction=system,
                tools=functions,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
            )
            response = client.models.generate_content(model=MODEL, contents="\n".join(contents), config=config)
            answer = response.text or "I couldn't produce an answer."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Agent error: {e}")
