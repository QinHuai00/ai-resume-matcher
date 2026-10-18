"""Streamlit 网页版（可选依赖；没装 streamlit 时打印安装提示后退出，不抛堆栈）。

运行：``streamlit run ui/app.py``

Streamlit 只有一个必须记住的坑：**任何交互都会从头重跑整个脚本**，所以分析结果
必须放进 ``st.session_state``，否则点一下按钮、上传一次文件结果就没了。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError:
    print(
        "未安装 streamlit，网页版不可用。\n"
        "安装后运行：pip install streamlit && streamlit run ui/app.py\n"
        "（离线模式下可以直接用命令行：python -m app.cli analyze --jd ... --resume ...）"
    )
    raise SystemExit(0) from None

from app import config  # noqa: E402
from app.llm import provider_info  # noqa: E402
from app.pipeline import analyze  # noqa: E402
from app.resume_parser import extract_text  # noqa: E402

st.set_page_config(page_title="AI 简历匹配助手（项目二初版）", page_icon="📄", layout="wide")
st.title("AI 简历—岗位匹配助手")
provider = provider_info()
st.caption(
    f"当前模式：{provider['mode']}（{provider['reason']}）｜ "
    f"Prompt 版本：{provider['prompt_version']}"
)

sample_jds = sorted(config.SAMPLES_DIR.glob("*.md"))
jd_choice = st.selectbox(
    "快速载入示例 JD（也可以直接粘贴自己的）",
    ["（手动粘贴）"] + [path.name for path in sample_jds],
)

col_jd, col_resume = st.columns(2)
with col_jd:
    default_jd = ""
    if jd_choice != "（手动粘贴）":
        default_jd = (config.SAMPLES_DIR / jd_choice).read_text(encoding="utf-8")
    jd_text = st.text_area(
        "岗位描述（JD）", value=default_jd, height=340, placeholder="把整段招聘说明粘进来…"
    )
with col_resume:
    uploaded = st.file_uploader(
        "上传简历（.docx / .pdf / .txt / .md）",
        type=["docx", "pdf", "txt", "md"],
        help="PDF 需要安装 pypdf；离线演示推荐上传 .md/.txt 或直接粘贴文本",
    )
    resume_paste = st.text_area(
        "或直接粘贴简历文本", height=200, placeholder="离线演示最快的方式：粘贴一栏式简历文本"
    )

if st.button("开始分析", type="primary"):
    if not jd_text.strip():
        st.warning("请先填写 JD")
        st.stop()

    resume_text = ""
    try:
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(uploaded.getbuffer())
                temp_path = Path(handle.name)
            try:
                resume_text = extract_text(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)
        elif resume_paste.strip():
            resume_text = resume_paste.strip()
        else:
            st.warning("请上传简历或粘贴简历文本")
            st.stop()

        with st.spinner("分析中（离线模式约 1 秒）…"):
            st.session_state["result"] = analyze(jd_text, resume_text)
    except Exception as exc:  # noqa: BLE001 —— 页面层兜住所有异常，给用户一句人话
        st.error(f"分析失败：{exc}")
        st.stop()

data = st.session_state.get("result")
if data:
    score = data["score"]
    col_total, col_must, col_skill, col_exp = st.columns(4)
    col_total.metric("总分", f"{score['total']}/100")
    col_must.metric("硬性要求命中", f"{score['must_hit_rate']:.0%}")
    col_skill.metric("技能覆盖", f"{score['skill_coverage']:.0%}")
    col_exp.metric("经历相关度", f"{score['exp_relevance']:.2f}")
    st.caption(
        f"缓存：{'命中' if data['meta']['cache_hit'] else '未命中'} ｜ "
        f"耗时 {data['meta']['elapsed_ms']} ms ｜ 模式 {data['meta']['mode']}"
    )
    with st.expander("完整 Markdown 报告", expanded=True):
        st.markdown(data["markdown"])
    st.download_button(
        "下载 Markdown 报告",
        data["markdown"],
        file_name="match_report.md",
        mime="text/markdown",
    )
