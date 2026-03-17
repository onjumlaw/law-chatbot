# ─────────────────────────────────────────────────────────────────
# 2025 양형기준 신호등 챗봇
# Gemini API + PDF 지식베이스 + Streamlit
# ─────────────────────────────────────────────────────────────────

import streamlit as st
import google.generativeai as genai
import pdfplumber
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일이 있으면 환경변수 자동 로드
load_dotenv()

# ─────────────────────────────────────────
# 페이지 기본 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="🚦 2025 양형기준 신호등 챗봇",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* 헤더 배경 */
    .main-header {
        text-align: center;
        padding: 1.2rem;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: white;
        border-radius: 12px;
        margin-bottom: 1.2rem;
    }
    /* 면책 박스 */
    .disclaimer {
        background: #fff8e1;
        border-left: 5px solid #ffa000;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.88rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 1. PDF 로드 함수
# ─────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_pdfs_from_folder(pdf_dir: str = "pdfs") -> tuple:
    """
    pdfs/ 폴더의 PDF 파일을 읽어 텍스트로 변환합니다.
    @st.cache_data 덕분에 같은 폴더는 앱 재실행 없이 한 번만 읽습니다.
    반환값: (전체 텍스트, 로드 성공 파일 수)
    """
    folder = Path(pdf_dir)
    folder.mkdir(exist_ok=True)

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        return "", 0

    all_texts = []
    loaded = 0
    for pdf_file in pdf_files:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                doc_text = f"\n\n{'━'*55}\n[문서] {pdf_file.name}\n{'━'*55}\n"
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        doc_text += f"\n--- p.{i} ---\n{page_text}\n"
                all_texts.append(doc_text)
                loaded += 1
        except Exception as e:
            # 로드 실패한 파일은 건너뜁니다
            all_texts.append(f"\n[오류: {pdf_file.name} 로드 실패 — {e}]\n")

    return "\n".join(all_texts), loaded


def load_pdfs_from_upload(uploaded_files) -> tuple:
    """사용자가 직접 업로드한 PDF 파일들을 텍스트로 변환합니다."""
    all_texts = []
    loaded = 0
    for f in uploaded_files:
        try:
            with pdfplumber.open(f) as pdf:
                doc_text = f"\n\n{'━'*55}\n[문서] {f.name}\n{'━'*55}\n"
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        doc_text += f"\n--- p.{i} ---\n{page_text}\n"
                all_texts.append(doc_text)
                loaded += 1
        except Exception as e:
            st.error(f"❌ {f.name} 처리 실패: {e}")

    return "\n".join(all_texts), loaded


# ─────────────────────────────────────────
# 2. 시스템 프롬프트 (챗봇 답변 알고리즘)
# ─────────────────────────────────────────

def build_system_prompt(knowledge_base: str) -> str:
    """
    Gemini 모델에게 전달할 '역할 및 답변 지침'을 생성합니다.
    knowledge_base에 PDF 텍스트가 있으면 참고 자료로 포함합니다.
    """
    # 너무 길면 앞 90,000자만 사용 (Gemini 토큰 한도 대비)
    MAX_KB = 90_000
    if len(knowledge_base) > MAX_KB:
        kb_text = knowledge_base[:MAX_KB] + "\n...[이하 분량 초과로 생략]"
    else:
        kb_text = knowledge_base

    if kb_text.strip():
        kb_section = f"""
[📚 양형기준 참고 자료 — PDF 원문]
{kb_text}
"""
    else:
        kb_section = """
[📚 양형기준 참고 자료]
※ PDF 자료가 제공되지 않았습니다. 대한민국 형법 및 일반적인 양형 실무를 기반으로 답변합니다.
"""

    return f"""당신은 대한민국 대법원 '2025 양형기준'을 기반으로 형사사건을 분석하는 전문 챗봇입니다.
일반 시민이 이해할 수 있도록 쉽고 친절하게 설명해 주세요.

{kb_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔵 답변 프로토콜 — 아래 단계를 반드시 순서대로 따르세요
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

◆ [첫 번째 메시지] 사용자가 사건 상황을 처음 설명할 때

## ⚖️ 1단계: 잠정 판단

> **예상 죄명:** [죄명]
> **관련 법조항:** [형법 제○조 등]
> **판단 근거:** [왜 이 죄에 해당하는지 2~3줄 설명. PDF 근거가 있으면 "[양형기준 참조]" 표시]

---

## ❓ 2단계: 정확한 분석을 위한 추가 질문

아래 항목 중 해당 사건에 **가장 중요한 2~3가지**를 골라 질문하세요.
- 초범 여부 (전과가 있는지)
- 피해자와의 합의 또는 피해 변제 여부
- 피해 금액 또는 피해 정도 (부상 정도 등)
- 자수 또는 자백 여부
- 반성 의사 표현 여부
- 공범 여부
- 범행 기간 및 횟수

**응답 형식 예시:**
"정확한 분석을 위해 몇 가지 여쭤보겠습니다:
1. [질문 1]
2. [질문 2]
3. [질문 3]"

---

◆ [두 번째 메시지 이후] 사용자가 추가 질문에 답변한 후

## 📊 3단계: 형량 범위 및 집행유예 검토

### 권고 형량 범위
| 구분 | 형량 |
|------|------|
| 기본 하한 | 예: 징역 6개월 |
| 기본 상한 | 예: 징역 2년 |
| 적용 유형 | 예: 사기범죄 제2유형 |

### 집행유예 가능성 분석
**✅ 긍정 요소 (감경 사유):**
- [요소 나열]

**❌ 부정 요소 (가중 사유):**
- [요소 나열]

**🔍 종합 의견:** [집행유예 가능성 수준 — 높음/보통/낮음 — 과 이유를 2~3줄로]

---

## 🚦 4단계: 항목별 양형 신호등

| 양형 요소 | 신호등 | 판단 근거 |
|-----------|--------|-----------|
| 합의 / 피해 회복 | 🟢/🟡/🔴 | [설명] |
| 전과 / 초범 여부 | 🟢/🟡/🔴 | [설명] |
| 반성 및 태도 | 🟢/🟡/🔴 | [설명] |
| 피해 규모 / 정도 | 🟢/🟡/🔴 | [설명] |
| 범행 수법 / 동기 | 🟢/🟡/🔴 | [설명] |

**🚦 종합 신호등:** [🟢 전반적으로 유리 / 🟡 유불리 혼재 / 🔴 전반적으로 불리]

> 🟢 유리한 요소 우세 &nbsp;|&nbsp; 🟡 혼재 &nbsp;|&nbsp; 🔴 불리한 요소 우세

---

⚠️ **면책 공지**
> 본 답변은 입력된 정보를 바탕으로 한 단순 조언이며, 구체적인 사안에 따라 결과가 달라질 수 있으므로 반드시 법률 전문가인 변호사와 상담하십시오.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ 핵심 지침
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 반드시 한국어로 답변하세요.
- 법률 용어는 쉬운 말로 풀어서 설명하세요.
- 형량은 범위로만 제시하고 "반드시", "확실히" 같은 단정 표현을 쓰지 마세요.
- 사용자가 추가 질문에 답변하기 전까지는 1단계와 2단계만 수행하세요.
- 사용자가 추가 정보를 제공한 후에 3단계와 4단계를 진행하세요.
- PDF 자료에 근거가 있으면 반드시 "[양형기준 참조]"를 표기하세요.
- 모르는 내용은 "일반적으로", "통상적으로" 등의 표현을 사용하세요.
"""


# ─────────────────────────────────────────
# 3. 세션 상태 초기화
# ─────────────────────────────────────────

def init_session():
    """앱 최초 실행 시 필요한 변수들을 초기화합니다."""
    defaults = {
        "messages": [],
        "chat_history": [],
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "system_prompt": build_system_prompt(""),
        "knowledge_base": "",
        "kb_loaded": False,
        "kb_char_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # ── 앱 시작 시 pdfs/ 폴더 자동 로드 ──
    if not st.session_state.kb_loaded:
        kb, count = load_pdfs_from_folder("pdfs")
        if count > 0:
            st.session_state.knowledge_base = kb
            st.session_state.system_prompt = build_system_prompt(kb)
            st.session_state.kb_loaded = True
            st.session_state.kb_char_count = len(kb)



# ─────────────────────────────────────────
# 4. Gemini API 호출
# ─────────────────────────────────────────

def get_ai_response(user_message: str, api_key: str, model_name: str) -> str:
    """
    Gemini 모델에 메시지를 보내고 응답 텍스트를 반환합니다.
    이전 대화 내역(chat_history)을 포함해 문맥을 유지합니다.
    """
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": 0.2,        # 낮을수록 일관된 답변 (법률 분석에 적합)
            "max_output_tokens": 2048,  # 충분한 응답 길이
        },
        system_instruction=st.session_state.system_prompt,
    )

    # 이전 대화 내역으로 채팅 세션 시작
    chat = model.start_chat(history=st.session_state.chat_history)
    response = chat.send_message(user_message)

    # 다음 메시지를 위해 대화 내역 업데이트
    st.session_state.chat_history.append({"role": "user",  "parts": [user_message]})
    st.session_state.chat_history.append({"role": "model", "parts": [response.text]})

    return response.text


# ─────────────────────────────────────────
# 5. 메인 앱
# ─────────────────────────────────────────

def main():
    init_session()

    # ── 상단 헤더 ──
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:1.8rem;">🚦 2025 양형기준 신호등 챗봇</h1>
        <p style="margin:0.4rem 0 0; opacity:0.85; font-size:0.95rem;">
            대법원 양형기준 기반 &nbsp;·&nbsp; 형량 예측 &nbsp;·&nbsp; 집행유예 분석
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ────────────────────────────────────────
    # 사이드바
    # ────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 설정")

        # ── API 키 입력 ──
        api_input = st.text_input(
            "🔑 Gemini API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="AIza...",
            help="Google AI Studio(aistudio.google.com)에서 무료 발급 가능",
        )
        if api_input != st.session_state.api_key:
            st.session_state.api_key = api_input

        # ── 모델 선택 ──
        model_name = st.selectbox(
            "🤖 Gemini 모델",
            options=["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"],
            index=0,
            help="flash: 빠르고 무료 할당량 넉넉 | pro: 더 정확하지만 속도 느림",
        )

        st.divider()

        # ── PDF 지식베이스 ──
        st.header("📚 양형기준 PDF")
        tab_folder, tab_upload = st.tabs(["폴더 로드", "직접 업로드"])

        with tab_folder:
            st.caption("프로젝트 폴더 안의 `pdfs/` 폴더에 PDF를 넣은 뒤 버튼을 누르세요.")
            if st.button("📂 pdfs/ 폴더 로드", use_container_width=True):
                with st.spinner("PDF 분석 중..."):
                    kb, count = load_pdfs_from_folder("pdfs")
                    if count > 0:
                        st.session_state.knowledge_base = kb
                        st.session_state.system_prompt = build_system_prompt(kb)
                        st.session_state.kb_loaded = True
                        st.session_state.kb_char_count = len(kb)
                        # 새 지식베이스 적용을 위해 대화 초기화
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.success(f"✅ {count}개 PDF 로드 완료!")
                        st.rerun()
                    else:
                        st.info("pdfs/ 폴더에 PDF 파일이 없습니다.\n파일을 넣고 다시 시도하세요.")

        with tab_upload:
            uploaded_files = st.file_uploader(
                "PDF 파일 선택",
                type="pdf",
                accept_multiple_files=True,
                help="여러 파일을 한 번에 선택할 수 있습니다.",
            )
            if uploaded_files:
                if st.button("📤 업로드 파일 적용", use_container_width=True):
                    with st.spinner("업로드 파일 처리 중..."):
                        kb, count = load_pdfs_from_upload(uploaded_files)
                        if count > 0:
                            st.session_state.knowledge_base = kb
                            st.session_state.system_prompt = build_system_prompt(kb)
                            st.session_state.kb_loaded = True
                            st.session_state.kb_char_count = len(kb)
                            st.session_state.messages = []
                            st.session_state.chat_history = []
                            st.success(f"✅ {count}개 파일 적용 완료!")
                            st.rerun()

        # 지식베이스 상태 표시
        if st.session_state.kb_loaded:
            st.success(f"📄 지식베이스 활성  \n{st.session_state.kb_char_count:,}자 로드됨")
        else:
            st.warning("📄 PDF 미로드  \n기본 법률 지식으로 답변합니다")

        st.divider()

        # ── 대화 초기화 ──
        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

        # ── 사용 안내 ──
        with st.expander("📖 사용 방법"):
            st.markdown("""
**순서:**
1. Gemini API 키 입력
2. 양형기준 PDF 로드 *(선택)*
3. 사건 상황을 자유롭게 입력
4. 챗봇 추가 질문에 답변
5. 형량 범위 + 신호등 확인

---
**PDF 구하는 곳:**
대법원 양형위원회 홈페이지
→ 양형기준 → 자료실

---
⚠️ **주의:** 본 챗봇은 참고용이며,
실제 사건은 변호사와 상담하세요.
            """)

    # ────────────────────────────────────────
    # 채팅 영역
    # ────────────────────────────────────────

    # 환영 메시지 (대화가 없을 때만 표시)
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🚦"):
            st.markdown("""
안녕하세요! **2025 양형기준 신호등 챗봇**입니다. 🚦

저는 대법원 양형기준을 바탕으로 **예상 형량 범위**와 **집행유예 가능성**을 분석해 드립니다.

**아래처럼 사건 상황을 자유롭게 말씀해 주세요:**

- "편의점에서 5만원어치 물건을 훔쳤어요"
- "술을 마시고 친구와 싸워서 코뼈를 부러뜨렸어요"
- "온라인 중고거래 사기로 300만원을 챙겼어요"
- "회사 돈 2,000만원을 몰래 빼돌렸어요"

> ⚠️ 본 챗봇은 **참고용**입니다.
> 실제 법률 문제는 반드시 **변호사**와 상담하세요.
            """)

    # 기존 대화 메시지 출력
    for msg in st.session_state.messages:
        avatar = "🙋" if msg["role"] == "user" else "🚦"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── 사용자 입력 처리 ──
    if user_input := st.chat_input("사건 상황을 설명하거나, 챗봇의 질문에 답변해 주세요..."):

        # API 키 확인
        if not st.session_state.api_key.strip():
            st.error("⚠️ 왼쪽 사이드바에서 **Gemini API 키**를 먼저 입력해주세요!")
            st.info("API 키는 Google AI Studio에서 무료로 발급받을 수 있습니다.")
            st.stop()

        # 사용자 메시지 저장 및 표시
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🙋"):
            st.markdown(user_input)

        # AI 응답 생성
        with st.chat_message("assistant", avatar="🚦"):
            with st.spinner("🔍 양형기준 분석 중..."):
                try:
                    response_text = get_ai_response(
                        user_message=user_input,
                        api_key=st.session_state.api_key,
                        model_name=model_name,
                    )
                    st.markdown(response_text)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                    })

                except Exception as e:
                    err = str(e)
                    if "API_KEY_INVALID" in err or "api key" in err.lower():
                        st.error("❌ API 키가 유효하지 않습니다. 키를 다시 확인해주세요.")
                    elif "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                        st.error("❌ API 사용량 초과입니다. 잠시 후 다시 시도해주세요.")
                    elif "SAFETY" in err or "blocked" in err.lower():
                        st.error("❌ 안전 필터에 의해 차단되었습니다. 다르게 표현해 주세요.")
                    elif "NOT_FOUND" in err or "not found" in err.lower():
                        st.error(f"❌ 모델을 찾을 수 없습니다. 다른 모델을 선택해보세요.\n오류: {err}")
                    else:
                        st.error(f"❌ 오류 발생: {err}")


if __name__ == "__main__":
    main()
