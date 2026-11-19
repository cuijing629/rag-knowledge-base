# RAG Knowledge Base

中文 PDF 知识库问答应用。使用 FastAPI、PyMuPDF、sentence-transformers、
ChromaDB、SQLite、Gemini 和 Streamlit。应用面向本地单实例演示。

## 安装

使用 Python 3.11 或 3.12 创建虚拟环境，并执行：

```text
python -m pip install -r requirements.txt
```

将 `.env.example` 复制到 `.env`，填写 `GEMINI_API_KEY`。
首次上传需要联网下载向量模型。

## 启动

在两个终端中激活同一个虚拟环境并进入项目根目录。

```text
python -m uvicorn backend.main:app
python -m streamlit run frontend/app.py
```

页面为 http://127.0.0.1:8501，交互式 API 为 http://127.0.0.1:8000/docs。

## 演示和测试

```text
python make_demo_pdf.py
python -m pytest -q
```

上传 `data/demo/实训手册.pdf`，提问「每个合并请求至少需要几名同学审查？」。
预期答案是至少两名，来源是第 2 页。测试使用模拟向量和 Gemini，
以及真实的临时 Chroma 和 SQLite。

## 架构

Streamlit 通过 HTTP 调用 FastAPI。PyMuPDF 逐页提取文本，分块后由本地模型向量化。
Chroma 按余弦距离检索，SQLite 保存文档状态和问答历史。Gemini 接收限长证据，
返回结构化答案和引用编号，后端将编号映射到真实原文和页码。

## API

- `GET /health` 健康检查
- `POST /api/documents/upload` 上传，multipart 字段为 `file`
- `GET /api/documents` 查询文档
- `DELETE /api/documents/{document_id}` 删除文档及向量
- `POST /api/chat` 提问，JSON 字段为 `question`、`session_id`，可选 `document_id`、`top_k`
- `GET /api/chat/history/{session_id}` 查询最近会话

本地索引和问答存放在 `data/`。发送问答时，有限的检索文本和会话历史会发给 Gemini。
更换向量模型时请使用新的 `DATA_DIR` 并重新导入。扫描 PDF 需要先做 OCR。
