import os
from uuid import uuid4

import httpx
import streamlit as st

API = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def call(method: str, path: str, **kwargs):
    try:
        response = httpx.request(method, API + path, timeout=180, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else True
    except httpx.HTTPStatusError as exc:
        try:
            message = exc.response.json().get("error", {}).get("message")
        except ValueError:
            message = None
        st.error(message or f"接口请求失败（HTTP {exc.response.status_code}）。")
    except httpx.TimeoutException:
        st.error("请求超时，可刷新列表检查索引是否完成。")
    except httpx.RequestError:
        st.error("无法连接后端。请检查 FastAPI 服务和 BACKEND_URL。")


def render(question, result):
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant"):
        st.text(result["answer"])
        if result["sources"]:
            with st.expander("查看引用来源"):
                for source in result["sources"]:
                    st.text(f"{source['filename']} · 第 {source['page']} 页")
                    st.text(source["excerpt"])
                    st.divider()


st.set_page_config(page_title="RAG Knowledge Base", page_icon="📚", layout="wide")
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid4().hex

with st.sidebar:
    st.header("文档库")
    with st.form("upload", clear_on_submit=True):
        pdf = st.file_uploader("选择 PDF", type=["pdf"])
        submitted = st.form_submit_button("上传并建立索引", type="primary")
    if submitted:
        if pdf is None:
            st.warning("请先选择文件。")
        else:
            with st.spinner("正在解析和建立索引…"):
                result = call("POST", "/api/documents/upload", files={
                    "file": (pdf.name, pdf.getvalue(), "application/pdf"),
                })
            if result:
                st.success(f"已索引 {result['chunks']} 个片段")
    documents = call("GET", "/api/documents")
    if documents == []:
        st.caption("文档库为空")
    for doc in documents or []:
        with st.container(border=True):
            st.text(doc["filename"])
            st.caption(f"{doc['pages_with_text']} 个文本页 · {doc['chunks']} 个片段")
            if st.button("删除", key=f"delete-{doc['id']}"):
                if call("DELETE", f"/api/documents/{doc['id']}"):
                    st.rerun()

st.title("RAG Knowledge Base")
st.caption("PDF 知识库问答")
if st.button("新建对话"):
    st.session_state.session_id = uuid4().hex
    st.rerun()

names = {doc["id"]: doc["filename"] for doc in documents or []}
scope = st.selectbox("检索范围", [None, *names], format_func=lambda key: names.get(key, "全部文档"))
session = st.session_state.session_id
history = call("GET", f"/api/chat/history/{session}")
for turn in history or []:
    render(turn["question"], turn)

if question := st.chat_input("输入关于 PDF 的问题", disabled=documents is None, max_chars=2000):
    with st.spinner("正在检索并生成回答…"):
        result = call("POST", "/api/chat", json={
            "question": question, "session_id": session, "document_id": scope,
        })
    if result:
        render(question, result)
