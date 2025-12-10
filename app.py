import streamlit as st
import random
import time
import cpx_engine

# 페이지 설정
st.set_page_config(page_title="CPX Simulator", page_icon="🏥", layout="wide")

# =========================================================
# [전역 설정] 데이터 로드 및 상태 초기화
# =========================================================
if "is_initialized" not in st.session_state:
    with st.spinner("AI 엔진 및 데이터 로딩 중..."):
        success = cpx_engine.initialize_data()
    st.session_state.is_initialized = success
    # 초기 모드는 'setup' (증상 선택 화면)
    st.session_state.mode = "setup"

# 데이터 로드 실패 시 중단
if not st.session_state.is_initialized:
    st.error("데이터 파일(json, faiss)을 찾을 수 없습니다. 폴더 위치를 확인해주세요.")
    st.stop()

# =========================================================
# [UI 1] 사이드바 (정보 패널) - 환자가 배정된 후에만 표시
# =========================================================
with st.sidebar:
    st.title("🏥 CPX Simulator")
    
    # 환자가 로드된 상태(채팅/평가/결과)일 때만 정보 표시
    if "patient" in st.session_state and st.session_state.mode != "setup":
        p = st.session_state.patient['profile']
        st.info(f"**환자:** {p['name']} ({p['age']}세 / {p['gender']})")
        
        with st.expander("🔍 정답지 (치트시트)"):
            st.json(st.session_state.patient['fact_sheet'])
            st.write(f"**Target:** {st.session_state.patient['target_disease']}")

        if st.button("🏠 처음으로 (증상 선택)"):
            st.session_state.mode = "setup"
            if "patient" in st.session_state:
                del st.session_state.patient
            st.rerun()
    else:
        st.write("👈 증상을 선택하고 시작하세요.")

# =========================================================
# [Mode 1] 설정 및 시작 화면 (Setup)
# =========================================================
if st.session_state.mode == "setup":
    st.markdown(
        """
        <div style='text-align: center; padding: 50px 0;'>
            <h1>🏥 의사 국가고시 실기(CPX) 시뮬레이터</h1>
            <p style='font-size: 1.2em; color: gray;'>
                AI 표준화 환자(SP)와 함께 실전처럼 문진을 연습해보세요.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # 1. 증상 선택 드롭다운
        symptoms_list = [
            "두통", "복통", "가슴통증", "기침", 
            "소변량 변화", "발열", "관절통증", "기분변화", "유방통"
        ]
        selected_symptom = st.selectbox("진료할 증상(Chief Complaint)을 선택하세요:", symptoms_list)

        st.write("") # 여백
        
        # 2. 시작하기 버튼
        if st.button("🚀 시뮬레이션 시작하기", use_container_width=True, type="primary"):
            
            # (중요) 선택한 증상에 맞는 환자 필터링 로직
            # 현재는 '두통' 데이터만 있으므로, '두통' 선택 시에만 로드하고 나머지는 경고 처리
            # 추후 데이터가 추가되면 여기서 필터링하면 됩니다.
            
            candidates = []
            
            # 임시 로직: 현재 json 파일이 두통 데이터라고 가정
            if selected_symptom == "두통":
                candidates = cpx_engine.scenarios
            else:
                # 만약 json 파일 내부에 'symptom': '복통' 같은 키가 있다면 아래 코드로 필터링 가능
                candidates = [p for p in cpx_engine.scenarios if p.get('symptom') == selected_symptom]

            if candidates:
                # 환자 랜덤 배정
                st.session_state.patient = random.choice(candidates)
                st.session_state.messages = []
                
                # 첫 인사 메시지 세팅
                opening = st.session_state.patient['opening_ment']
                st.session_state.messages.append({"role": "assistant", "content": opening})
                
                # 모드 전환 -> 채팅
                st.session_state.mode = "chat"
                st.rerun()
            else:
                st.warning(f"⚠️ 현재 '{selected_symptom}'에 대한 환자 데이터가 준비되지 않았습니다. ('두통'을 선택해보세요)")


# =========================================================
# [Mode 2] 채팅 모드 (Chat)
# =========================================================
elif st.session_state.mode == "chat":
    st.subheader(f"💬 문진 진행 중: {st.session_state.patient['profile']['name']} 님")
    st.caption("'진료종료' 또는 '그만'을 입력하면 답안 작성 단계로 넘어갑니다.")

    # 대화 기록 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 입력창
    if prompt := st.chat_input("의사로서 질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 🛑 종료 감지
        if any(w in prompt.lower() for w in ['종료', '그만', '끝', 'quit', '진료종료']):
            st.session_state.mode = "assessment"
            st.rerun()
        
        # 엔진 호출
        else:
            with st.chat_message("assistant"):
                with st.spinner("생각 중..."):
                    sel_id, reply = cpx_engine.search_and_process(st.session_state.patient, prompt)
                st.write(reply)
            
            st.session_state.messages.append({"role": "assistant", "content": reply})

# =========================================================
# [Mode 3] 답안 작성 모드 (Assessment)
# =========================================================
elif st.session_state.mode == "assessment":
    st.subheader("📝 진료가 종료되었습니다. 답안을 작성하세요.")
    
    with st.form("cpx_form"):
        st.markdown("### 감별 진단 (Differential Diagnosis)")
        
        c1, c2 = st.columns(2)
        with c1:
            dx1 = st.text_input("1순위 진단명 (필수)", placeholder="가장 의심되는 질환")
            dx2 = st.text_input("2순위 진단명")
            dx3 = st.text_input("3순위 진단명")
        with c2:
            plan1 = st.text_input("1순위 검사/치료 계획", placeholder="확진을 위한 검사")
            plan2 = st.text_input("2순위 계획")
            plan3 = st.text_input("3순위 계획")

        if st.form_submit_button("제출 및 채점 받기", use_container_width=True):
            user_answers = [
                {'rank': 1, 'dx': dx1, 'plan': plan1},
                {'rank': 2, 'dx': dx2, 'plan': plan2},
                {'rank': 3, 'dx': dx3, 'plan': plan3}
            ]
            
            with st.spinner("채점관(교수님)이 대화 내용을 분석하고 점수를 매기는 중입니다..."):
                # 여기에 st.session_state.messages를 추가로 전달합니다!
                feedback = cpx_engine.evaluate_assessment(st.session_state.patient, user_answers, st.session_state.messages)
            
            st.session_state.feedback = feedback
            st.session_state.mode = "result"
            st.rerun()

# =========================================================
# [Mode 4] 결과 확인 모드 (Result)
# =========================================================
elif st.session_state.mode == "result":
    st.balloons()
    st.title("📊 채점 결과")
    
    st.markdown(st.session_state.feedback)
    
    if st.button("새로운 시뮬레이션 시작하기 (초기 화면)"):
        st.session_state.mode = "setup"
        del st.session_state.patient

        st.rerun()

