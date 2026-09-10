import os
import re
import io
import pandas as pd
import requests
import streamlit as st
from google import genai
from google.genai import types


st.set_page_config(
    page_title="AGENTX",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# TOOL 1: Uploaded file reader
# -----------------------------
def read_uploaded_file(file_name: str, file_content: str) -> str:
    """
    Analyze an uploaded CSV or Excel file.
    Use this tool when the user asks questions about an uploaded file.
    """

    try:
        if file_name.lower().endswith(".csv"):
            df = pd.read_csv(io.StringIO(file_content))
        else:
            df = pd.read_excel(io.BytesIO(file_content.encode("latin1")))

        summary = {
            "file_name": file_name,
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "first_rows": df.head(10).to_dict(orient="records"),
        }

        return str(summary)

    except Exception as e:
        return f"Error reading file: {e}"


# -----------------------------
# TOOL 2: Google Sheets reader
# -----------------------------
def read_google_sheet(sheet_url: str) -> str:
    """
    Read a publicly accessible Google Sheet and return useful information.
    The sheet must be shared as 'Anyone with the link'.
    """

    try:
        match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)

        if not match:
            return "Invalid Google Sheets URL."

        sheet_id = match.group(1)

        csv_url = (
            f"https://docs.google.com/spreadsheets/d/"
            f"{sheet_id}/export?format=csv"
        )

        response = requests.get(csv_url, timeout=15)

        if response.status_code != 200:
            return "Could not access the Google Sheet. Make sure it is public."

        df = pd.read_csv(io.StringIO(response.text))

        return str({
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "first_rows": df.head(10).to_dict(orient="records")
        })

    except Exception as e:
        return f"Error reading Google Sheet: {e}"


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ AGENTX")

api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file:
    file_bytes = uploaded_file.getvalue()

    try:
        if uploaded_file.name.endswith(".csv"):
            uploaded_df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            uploaded_df = pd.read_excel(io.BytesIO(file_bytes))

        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.session_state["uploaded_file_data"] = uploaded_df.to_csv(index=False)

        st.sidebar.success(
            f"Loaded: {uploaded_file.name}"
        )

    except Exception as e:
        st.sidebar.error(f"File error: {e}")


# -----------------------------
# Main UI
# -----------------------------
st.title("🤖 AGENTX")
st.subheader("Multi-Tool AI Agent")

st.write(
    "An AI agent that can reason, use tools, analyze files "
    "and access public Google Sheets."
)


if not api_key:
    st.info("👈 Enter your Gemini API key in the sidebar to start.")
    st.stop()


# -----------------------------
# Gemini client
# -----------------------------
client = genai.Client(api_key=api_key)


system_instruction = """
You are AGENTX, an intelligent multi-tool AI assistant.

You can:
1. Have normal conversations.
2. Analyze uploaded CSV and Excel files.
3. Read publicly accessible Google Sheets.
4. Decide when a tool is needed.
5. Explain your results clearly.

When a user asks about an uploaded file, use the file analysis tool.
When a user provides a Google Sheets URL and asks about its data,
use the Google Sheets tool.

Do not claim to have used a tool if you did not use it.
"""


# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Chat input
# -----------------------------
user_prompt = st.chat_input(
    "Ask AGENTX anything..."
)


if user_prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Prepare context about uploaded file
    extra_context = ""

    if "uploaded_file_data" in st.session_state:
        extra_context = f"""
An uploaded file is available.

File name:
{st.session_state.get("uploaded_file_name")}

File data:
{st.session_state.get("uploaded_file_data")}
"""

    # Detect Google Sheet URL
    sheet_url_match = re.search(
        r"https?://docs\.google\.com/spreadsheets/[^\s]+",
        user_prompt
    )

    with st.chat_message("assistant"):

        try:

            # Tool: uploaded file
            if (
                "uploaded_file_data" in st.session_state
                and any(
                    word in user_prompt.lower()
                    for word in [
                        "file",
                        "csv",
                        "excel",
                        "spreadsheet",
                        "uploaded",
                        "data"
                    ]
                )
            ):

                result = read_uploaded_file(
                    st.session_state["uploaded_file_name"],
                    st.session_state["uploaded_file_data"]
                )

                prompt = f"""
{system_instruction}

User request:
{user_prompt}

Tool used: Uploaded File Analyzer

Tool result:
{result}

Answer the user using the tool result.
"""

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                answer = response.text

                st.markdown("🔧 **Tool used:** Uploaded File Analyzer")
                st.markdown(answer)

            # Tool: Google Sheets
            elif sheet_url_match:

                sheet_url = sheet_url_match.group(0)

                result = read_google_sheet(sheet_url)

                prompt = f"""
{system_instruction}

User request:
{user_prompt}

Tool used: Google Sheets Analyzer

Tool result:
{result}

Answer the user using the tool result.
"""

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                answer = response.text

                st.markdown("🔧 **Tool used:** Google Sheets Analyzer")
                st.markdown(answer)

            # Normal AI conversation
            else:

                conversation = "\n".join(
                    [
                        f"{m['role']}: {m['content']}"
                        for m in st.session_state.messages[-10:]
                    ]
                )

                prompt = f"""
{system_instruction}

Conversation:
{conversation}

{extra_context}

User's latest request:
{user_prompt}
"""

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                answer = response.text

                st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:

            error_message = f"""
❌ **Agent error**

`{str(e)}`

Please check your Gemini API key and model availability.
"""

            st.error(error_message)
