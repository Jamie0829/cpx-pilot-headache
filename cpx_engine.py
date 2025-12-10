import os
import json
import time
import re
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# =========================================================
# [전역 변수 설정]
# =========================================================
embedder = None
index = None
id_map = {}
scenarios = []
id_to_text_map = {}
client = None

# =========================================================
# 1. 초기화 함수 (데이터 로드) - 경로 문제 해결 버전
# =========================================================
def initialize_data():
    global embedder, index, id_map, scenarios, id_to_text_map, client
    
    print("⏳ 데이터 로딩 및 초기화 중...")

    # [핵심 수정] 현재 실행 중인 파일(cpx_engine.py)의 절대 경로를 기준점으로 잡습니다.
    # 이렇게 하면 서버의 현재 작업 폴더가 어디든 상관없이 항상 올바른 위치를 찾습니다.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 1. API 키 로드
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ .env 파일 또는 Secrets에서 OPENAI_API_KEY를 찾을 수 없습니다.")
        return False
        
    client = OpenAI(api_key=api_key)
    
    # 2. 임베딩 모델 로드
    try:
        embedder = SentenceTransformer('jhgan/ko-sroberta-multitask')
    except Exception as e:
        print(f"❌ 임베딩 모델 로드 실패: {e}")
        return False
    
    # 3. 데이터 파일 로드
    try:
        # [수정] 절대 경로 생성
        faiss_path = os.path.join(BASE_DIR, 'headache.faiss')
        meta_path = os.path.join(BASE_DIR, 'headache_meta.json')
        scenarios_path = os.path.join(BASE_DIR, 'headache_scenarios.json')
        master_path = os.path.join(BASE_DIR, 'headache_master.json')

        # FAISS 인덱스 로드
        if os.path.exists(faiss_path):
            index = faiss.read_index(faiss_path)
        else:
            print(f"❌ 파일 없음: {faiss_path}")
            return False

        # 메타 데이터 로드
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                id_map = json.load(f)
        else:
            print(f"❌ 파일 없음: {meta_path}")
            return False

        # 시나리오 데이터 로드
        if os.path.exists(scenarios_path):
            with open(scenarios_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                scenarios = data.get('scenarios', [])
        else:
            print(f"❌ 파일 없음: {scenarios_path}")
            return False
            
        # 마스터 데이터 로드 (Re-ranking용)
        if os.path.exists(master_path):
            with open(master_path, 'r', encoding='utf-8') as f:
                master_data = json.load(f)
            # id_to_text_map이 전역 변수로 선언되어 있다고 가정하고 업데이트
            if 'id_to_text_map' not in globals():
                id_to_text_map = {} 
            for item in master_data.get('checklist', []):
                id_to_text_map[item['id']] = item['standard_text']
        else:
            print(f"⚠️ 파일 없음: {master_path} (질문 매칭 정확도 하락 가능)")
            
        print("✅ 모든 데이터 로드 완료")
        return True

    except Exception as e:
        print(f"❌ 데이터 초기화 중 오류 발생: {e}")
        return False

# =========================================================
# 2. GPT 호출 헬퍼 (max_tokens 파라미터 지원)
# =========================================================
def generate_gpt(prompt, model="gpt-4o", max_tokens=300):
    if not client: return "API Client Error"
    
    max_retries = 3
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "당신은 의학 시뮬레이션의 표준화 환자(SP) 혹은 채점관입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7, 
                max_tokens=max_tokens 
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "RateLimit" in str(e):
                time.sleep(1)
            else:
                return f"Error: {e}"
    return "응답 생성 실패"

# =========================================================
# 3. 환자 엔진 (대화 처리)
# =========================================================
def search_and_process(patient, user_text):
    # 1. 후보군 추출 (Top 10)
    vector = embedder.encode([user_text])
    faiss.normalize_L2(vector)
    similarities, indices = index.search(vector, 10) 
    
    candidates_list_str = ""
    for i in range(10):
        q_id = id_map[indices[0][i]]
        std_text = id_to_text_map.get(q_id, "내용 미상")
        candidates_list_str += f"- {q_id}: {std_text}\n"

    # 2. GPT Re-ranking & Acting (페르소나 적용)
    profile = patient['profile']
    
    prompt = f"""
    [역할] {profile['name']} ({profile['age']}세, {profile['job']}), 의학 지식 없음.
    [질문] "{user_text}"
    
    [1단계: 의도 매칭]
    아래 후보 중 질문과 가장 일치하는 ID를 고르세요.
    {candidates_list_str}
    
    [2단계: 답변 생성]
    선택한 ID의 [Fact]를 찾아 환자 말투로 연기하세요.
    [Fact Data]: {json.dumps(patient['fact_sheet'], ensure_ascii=False)}
    
    ★ [말투 가이드라인]
    1. **의학 용어 금지**: "발열"->"열나요", "오한"->"추워요"
    2. **구어체/감정**: "아니요"->"아뇨, 없었어요.", 아픈 소리(...으으..) 포함.
    3. **데이터 부재 시**: "잘 모르겠어요" 또는 "기억 안 나요". (지어내기 금지)
    
    [출력 포맷]
    ID || 환자대사
    """
    
    raw_response = generate_gpt(prompt, max_tokens=300)
    
    sel_id = "Unknown"
    reply = "죄송해요, 다시 말씀해주시겠어요?"
    
    try:
        if raw_response and "||" in raw_response:
            parts = raw_response.split("||")
            sel_id = parts[0].strip()
            reply = parts[1].strip()
            # ID 정제
            sel_id = re.sub(r'^[0-9]+[\.\-\)\s]*', '', sel_id).replace("[", "").replace("]", "").strip()
        elif raw_response:
            reply = raw_response
    except: pass

    # 3. 강제 부정 로직 (데이터 없을 때 거짓말 방지)
    real_fact = patient['fact_sheet'].get(sel_id)
    if not real_fact and sel_id != "Unknown" and sel_id != "Empathy":
        if any(k in sel_id for k in ['History', 'Experience', 'AssociatedSx']):
            reply = "아뇨, 그런 건 딱히 없었어요."
        else:
            reply = "글쎄요... 정확히는 잘 기억이 안 나네요."
            
    return sel_id, reply

# =========================================================
# 4. 채점 함수 (평가 로직)
# =========================================================
def evaluate_assessment(patient, user_answers, chat_history):
    """
    Step 4: 종합 채점 
    (1) 병력청취 (Demographics, PE 제외)
    (2) 환자교육 (10점)
    (3) PPI (18점 표)
    (4) 진단/계획
    """
    true_dx = patient['target_disease']
    true_plan = ", ".join(patient['diagnostic_plan'])
    
    # ---------------------------------------------------------
    # 🛠️ [채점 기준 필터링]
    # - 값이 비어있는 항목 제외
    # - 'KQ_Demographics'로 시작하는 항목 제외
    # - 'PE'로 시작하는 항목 제외
    # ---------------------------------------------------------
    checklist_items = []
    for k, v in patient['fact_sheet'].items():
        if not v: continue # 값 없음 제외
        if k.startswith('KQ_Demographics'): continue # 인적사항 제외
        if k.startswith('PE'): continue # 신체진찰 제외
        
        checklist_items.append(f"- {k} (내용: {v})")

    checklist_str = "\n".join(checklist_items)

    # 학생 답안 포맷팅
    student_dx_plan = ""
    for item in user_answers:
        dx = item['dx'] if item['dx'] else "(입력 안 함)"
        plan = item['plan'] if item['plan'] else "(입력 안 함)"
        student_dx_plan += f"- [{item['rank']}순위] 진단: {dx} | 검사: {plan}\n"

    # 대화 기록 포맷팅
    transcript = ""
    for msg in chat_history:
        role = "의사" if msg['role'] == "user" else "환자"
        transcript += f"{role}: {msg['content']}\n"

    # 메가 프롬프트 작성
    prompt = f"""
    당신은 의사 국가고시 실기(CPX) 수석 채점관입니다.
    제공된 자료를 바탕으로 4가지 항목을 채점하여 성적표를 작성하세요.

    [상황 정보]
    - 정답 진단: {true_dx}
    - 필수 검사: {true_plan}
    
    [채점 대상 필수 문진 항목 (Checklist)]
    {checklist_str}

    [진료 대화 기록]
    {transcript}

    [학생 답안]
    {student_dx_plan}

    ---
    [채점 기준 및 출력 양식 (Markdown)]
    
    # 📊 CPX 종합 성적표

    ## 1. 병력 청취 (History Taking)
    * **채점 방식**: 위 [채점 대상 필수 문진 항목]에 있는 정보들을 의사가 질문을 통해 알아냈는지 확인하세요.
    * **주의**: 질문을 하지 않아도 환자가 스스로 말한 정보는 획득한 것으로 인정합니다.
    * **출력 형식**: 
      - 체크리스트 표 (항목명 | 획득여부(O/X) | 배점(1점))
      - **총점**: (획득점수) / (총 항목 수) 점

    ## 2. 환자 교육 (Patient Education)
    * **채점 기준**: 진료 후반부에 환자에게 현재 상태(진단), 검사 계획, 생활 습관 교정 등을 설명했는가?
    * **출력 형식**:
      - 교육 여부: (있음/없음)
      - 내용 충실도 평가: (교육 내용 요약 및 평가 1문장)
      - **점수**: (0~10점) / 10점 만점

    ## 3. 의사-환자 관계 (PPI)
    * **채점 기준**: 아래 6개 항목 평가 (3:아주우수, 2:우수, 1:보통, 0:미흡)
    * **출력 형식**: (아래 표 작성)
    | 평가 항목 | 점수 (0~3) | 평가 근거 |
    |---|---|---|
    | 1. 효율적인 병력 청취 | | |
    | 2. 경청 및 공감적 태도 | | |
    | 3. 이해하기 쉬운 설명 | | |
    | 4. 환자의 의견 존중 | | |
    | 5. 비언어적 소통(태도) | | |
    | 6. 신뢰감 및 전문성 | | |
    * **PPI 총점**: (합계) / 18점

    ## 4. 진단 및 계획 (Assessment & Plan)
    * **진단 정확도**: (정답 진단 포함 여부 O/X)
    * **검사 적절성**: (필수 검사 포함 여부 O/X)
    * **피드백**: (의학적 조언)

    ## 🏆 최종 총평
    > (학생에게 전하는 조언)
    """
    
    # 긴 리포트를 위해 토큰 제한 늘림

    return generate_gpt(prompt, max_tokens=3000)
