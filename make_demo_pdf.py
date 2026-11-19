from pathlib import Path

import pymupdf


path = Path("data/demo/实训手册.pdf")
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    raise SystemExit(f"演示文件已存在：{path}")

with pymupdf.open() as pdf:
    page = pdf.new_page()
    page.insert_text((60, 80), "实训作业每周五 18:00 截止。", fontname="china-s", fontsize=16)
    page = pdf.new_page()
    page.insert_text((60, 80), "每个合并请求至少需要两名同学审查。", fontname="china-s", fontsize=16)
    pdf.save(path)

print(path)
