"""
ThinkPRM Confidence Score Generation Script (OpenAI Ver.)

thinkprm_data.json의 각 검증 스텝에 confidence score를 추가하여
thinkprm_data_conf.json을 생성합니다.

사용법:
1. OPENAI_API_KEY 환경 변수를 설정합니다.
   export OPENAI_API_KEY='your_api_key_here'

2. 터미널에서 이 스크립트 실행:
   python add_conf.py
"""

import json
import os
import sys
import re
import openai  # SGLang 대신 OpenAI 임포트
from tqdm import tqdm
import traceback

# ============================================================================
# 설정
# ============================================================================
# SGLANG_ENDPOINT = "http://127.0.0.1:31111" # SGLang 제거
# MODEL_NAME_OR_PATH = "KirillR/QwQ-32B-Preview-AWQ" # 로컬 모델 제거

# OpenAI 설정 추가
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_KEY = ""
OPENAI_MODEL_NAME = "gpt-5-nano" # 요청하신 모델명

N_SAMPLES_PER_STEP = 10  # 각 스텝당 생성할 검증 CoT 개수
MAX_GENERATION_TOKENS = 2048
INPUT_FILENAME = "thinkprm_data.json"
OUTPUT_FILENAME = "thinkprm_data_conf.json"
DEBUG_LOG_FILENAME = "add_conf_debug.log"

# SGLang의 stop_patterns를 상수로 이동
STOP_PATTERNS = [
    "\n\nStep ",  # 다음 스텝이 시작되면 중단
    "<|im_end|>",
    "</s>",
    "</think>"
]

# ============================================================================
# 로깅 함수
# ============================================================================

log_file = None

def log(message):
    """콘솔과 파일에 동시 출력"""
    print(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()

# ============================================================================
# 유틸리티 함수
# ============================================================================

def is_verification_chunk(chunk):
    """검증 청크인지 확인 (Step k:로 시작하고 \\boxed{}로 끝나는지)"""
    chunk = chunk.strip()
    if not chunk.startswith("Step"):
        return False
    if "\\boxed{" not in chunk:
        return False
    return True

def get_prefix_steps_until(prefix_steps, step_index):
    """
    step_index번째 스텝까지의 prefix_steps만 반환
    
    Args:
        prefix_steps: 전체 prefix_steps 리스트
        step_index: 포함하고 싶은 마지막 스텝 인덱스 (0-based)
    
    Returns:
        step_index번째 스텝까지의 prefix_steps를 '\n'으로 결합한 문자열
    """
    # step_index는 0-based이므로, step_index+1개까지 포함
    selected_steps = prefix_steps[:step_index + 1]
    return '\n'.join(selected_steps)

def get_prefix_before_step(cot_chunks, step_index):
    """
    step_index번째 검증 청크 직전까지의 검증 청크들만 연결
    단, 맨 앞의 "<think>\n"은 포함하되, 첫 검증 청크 이전의 다른 텍스트는 제외
    
    Args:
        cot_chunks: 전체 cot_chunks 리스트
        step_index: 검증하려는 스텝의 인덱스 (0-based)
    
    Returns:
        "<think>\n" + 검증 청크들만 연결한 문자열
    """
    prefix_chunks = []
    verification_count = 0
    first_verification_found = False
    
    for chunk in cot_chunks:
        if is_verification_chunk(chunk):
            # 첫 번째 검증 청크를 발견한 순간부터 수집 시작
            first_verification_found = True
            
            if verification_count == step_index:
                # 목표 검증 청크에 도달하면 중단
                break
            verification_count += 1
            prefix_chunks.append(chunk)
        elif first_verification_found:
            # 첫 검증 청크 발견 이후의 텍스트만 수집 (청크 사이의 "\n\n" 등)
            prefix_chunks.append(chunk)
    
    # 맨 앞에 "<think>\n" 추가
    return "<think>\n" + ''.join(prefix_chunks)

def extract_boxed_label(text):
    """생성된 텍스트에서 \\boxed{correct} 또는 \\boxed{incorrect} 추출"""
    pattern = r'\\boxed\{(correct|incorrect)\}'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        label = match.group(1).lower()
        return 1 if label == "correct" else 0
    return None

def format_verification_prompt(problem, prefix_steps_until_current, cot_prefix, current_step_number):
    """
    OpenAI API에 맞는 메시지 리스트 생성 (수정된 버전)
    
    Args:
        problem: 수학 문제
        prefix_steps_until_current: 현재 검증할 스텝까지의 풀이 과정
        cot_prefix: 이전 검증 청크들 (컨텍스트로 사용)
        current_step_number: 현재 검증할 스텝 번호 (1-based)
    """
    
    # 이전 검증 내용(cot_prefix)을 컨텍스트로 가공
    previous_verifications_text = ""
    if cot_prefix == "<think>\n":
        # Step 1을 검증할 차례 (이전 검증 내용 없음)
        previous_verifications_text = "You have not verified any steps yet."
    else:
        # Step 2 이상일 경우, <think> 태그를 제외한 내용만 추출
        previous_verifications_text = cot_prefix[8:].strip()
        
    
    # 새 지시사항이 포함된 User 프롬프트
    user_content = f"""[Problem]
{problem}

[Solution Steps]
{prefix_steps_until_current}

[Previous Verifications]
Below are the verifications you have already completed.
{previous_verifications_text}

[Your Task]
You must now verify **ONLY Step {current_step_number}**.
Provide a brief critique for Step {current_step_number} and then conclude with EXACTLY one of:
- The step is \\boxed{{correct}} (if the step is correct)
- The step is \\boxed{{incorrect}} if the step contains an error)

Do not repeat verifications for previous steps.

Output:"""

    messages = [
        {
            "role": "system",
            "content": "You are a mathematical reasoning verification assistant."
        },
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": "<think>\n" # 모델이 CoT를 시작하도록 유도
        }
    ]
    
    return messages

# ============================================================================
# SGLang 생성 함수 (제거)
# ============================================================================
# prompt_to_states = {}
# @function
# def generate_step_verification(s, prompt: str, n: int):
# ... (SGLang 함수 전체 제거)

# ============================================================================
# 메인 함수
# ============================================================================

def main():
    global log_file
    
    # 로그 파일 열기
    log_file = open(DEBUG_LOG_FILENAME, 'w', encoding='utf-8')
    
    log("=" * 70)
    log("ThinkPRM Confidence Score Generation (OpenAI Ver.)")
    log("=" * 70)
    
    # 1. OpenAI 클라이언트 초기화
    log(f"\n[1/5] OpenAI 클라이언트 초기화 중...")
    
    if not OPENAI_API_KEY:
        log("    ❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        log_file.close()
        return 1
    
    try:
        # OpenAI 클라이언트 인스턴스 생성
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        log(f"    ✓ OpenAI 클라이언트 초기화 성공 (Model: {OPENAI_MODEL_NAME})")
    except Exception as e:
        log(f"    ❌ OpenAI 클라이언트 초기화 실패: {e}")
        log_file.close()
        return 1

    # 2. 토크나이저 로드 (제거)
    # log(f"\n[2/5] 토크나이저 로드 중...")
    
    # 3. 입력 데이터 로드 (섹션 번호 변경 [2/5])
    log(f"\n[2/5] 입력 데이터 로드 중...")
    log(f"    파일: {INPUT_FILENAME}")
    
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data = data[:10]
        
        log(f"    ✓ 로드 성공: {len(data):,}개 문제")
        
        total_steps_to_verify = sum(item['valid_prefix_step_count'] for item in data)
        total_api_calls = total_steps_to_verify * N_SAMPLES_PER_STEP
        log(f"    - 검증할 총 스텝 수: {total_steps_to_verify:,}개")
        log(f"    - 예상 API 호출 횟수: {total_api_calls:,}회 (스텝당 {N_SAMPLES_PER_STEP}회)")
        
    except Exception as e:
        log(f"    ❌ 로드 실패: {e}")
        log_file.close()
        return 1

    # 4. Confidence Score 생성 (섹션 번호 변경 [3/5])
    log(f"\n[3/5] Confidence Score 생성 중...")
    log(f"    - 문제 수: {len(data)}")
    log(f"    - 스텝당 샘플링 횟수: {N_SAMPLES_PER_STEP}\n")
    
    result_data = []
    total_steps_processed = 0
    n_skipped = 0
    
    for problem_idx, item in enumerate(tqdm(data, desc="문제 처리 중")):
        try:
            problem = item['problem']
            prefix = item['prefix']
            prefix_steps = item['prefix_steps']
            cot_chunks = item['cot_chunks']
            gt_step_labels = item['gt_step_labels']
            valid_prefix_step_count = item['valid_prefix_step_count']
            
            log(f"\n{'=' * 70}")
            log(f"문제 {problem_idx}")
            log(f"{'=' * 70}")
            log(f"문제 텍스트:\n{problem}")
            log(f"\nvalid_prefix_step_count: {valid_prefix_step_count}")
            log(f"gt_step_labels: {gt_step_labels}")
            
            # 업데이트할 cot_chunks 복사
            updated_cot_chunks = cot_chunks.copy()
            
            # valid_prefix_step_count만큼만 처리
            for step_idx in range(valid_prefix_step_count):
                log(f"\n{'-' * 70}")
                log(f"Step {step_idx + 1} 처리 시작")
                log(f"{'-' * 70}")
                
                # 현재 스텝까지의 prefix_steps 가져오기
                prefix_steps_until_current = get_prefix_steps_until(prefix_steps, step_idx)
                
                log(f"\n=== Prefix Steps (Step {step_idx + 1}까지) ===")
                log(prefix_steps_until_current)
                log(f"=== Prefix Steps 끝 ===\n")
                
                # 해당 스텝 직전까지의 CoT 프리픽스 생성
                cot_prefix = get_prefix_before_step(cot_chunks, step_idx)
                
                log(f"\n=== CoT Prefix (Step {step_idx}까지 검증) ===")
                log(cot_prefix)
                log(f"=== CoT Prefix 끝 (길이: {len(cot_prefix)} 문자) ===\n")
                
                # 프롬프트 생성 (OpenAI API 형식)
                messages = format_verification_prompt(
                    problem=problem,
                    prefix_steps_until_current=prefix_steps_until_current,
                    cot_prefix=cot_prefix,
                    current_step_number=(step_idx + 1)  # 👈 이 줄 추가
                )
                
                log(f"API 요청 메시지 생성 완료\n")
                
                # prompt_to_states 초기화 (제거)
                # global prompt_to_states
                # prompt_to_states = {}
                
                # N_SAMPLES_PER_STEP번 생성 (10번 개별 호출)
                log(f"API 요청 시작 (총 {N_SAMPLES_PER_STEP}번 개별 호출)...\n")
                
                generated_verifications = []
                
                for i in range(N_SAMPLES_PER_STEP):
                    log(f"    -> 샘플 {i + 1}/{N_SAMPLES_PER_STEP} 요청 중...")
                    
                    try:
                        response = client.chat.completions.create(
                            model=OPENAI_MODEL_NAME,
                            messages=messages,
                            max_completion_tokens=MAX_GENERATION_TOKENS,
                            temperature=1.0,
                            #stop=STOP_PATTERNS,
                            n=1  # 👈 한 번에 1개만 요청
                        )
                        
                        # 생성된 결과 수집
                        verification = response.choices[0].message.content
                        generated_verifications.append(verification.strip())
                    
                    except Exception as api_err:
                        log(f"    ❌ 샘플 {i + 1} 생성 중 오류 발생: {api_err}")
                        log(f"    ⚠️ 해당 샘플은 건너뜁니다.")
                        # 이 샘플은 건너뛰고 다음 루프 계속 진행
                        continue 
                
                log(f"\n총 {len(generated_verifications)}/{N_SAMPLES_PER_STEP}개 샘플 수집 완료.")
                
                # GT 라벨과 비교
                gt_label = gt_step_labels[step_idx]
                gt_numeric = 1 if gt_label == '+' else 0
                
                log(f"GT 라벨: {gt_label} ({gt_numeric})\n")
                log(f"{'=' * 70}")
                log(f"=== 생성된 검증들 ===")
                log(f"{'=' * 70}")
                
                # 각 생성 결과에서 라벨 추출
                predicted_labels = []
                for i, gen_text in enumerate(generated_verifications):
                    log(f"\n[샘플 {i+1}]")
                    log("-" * 70)
                    log(gen_text)
                    log("-" * 70)
                    
                    pred_label = extract_boxed_label(gen_text)
                    predicted_labels.append(pred_label)
                    log(f"추출된 라벨: {pred_label} ({'correct' if pred_label == 1 else 'incorrect' if pred_label == 0 else 'PARSE_FAILED'})")
                
                # Confidence 계산 (일치 비율)
                valid_predictions = [p for p in predicted_labels if p is not None]
                
                log(f"\n{'=' * 70}")
                log(f"=== 결과 분석 ===")
                log(f"{'=' * 70}")
                
                if len(valid_predictions) > 0:
                    matches = sum(1 for pred in valid_predictions if pred == gt_numeric)
                    confidence = matches / len(valid_predictions)
                    log(f"총 생성: {len(predicted_labels)}개")
                    log(f"파싱 성공: {len(valid_predictions)}개")
                    log(f"GT와 일치: {matches}개")
                    log(f"Confidence: {confidence:.2f}")
                else:
                    confidence = 0.0
                    log(f"⚠️ 파싱 성공한 예측 없음!")
                
                # 해당 검증 청크에 confidence 태그 추가
                verification_count = 0
                chunk_found = False
                for chunk_idx, chunk in enumerate(updated_cot_chunks):
                    if is_verification_chunk(chunk):
                        if verification_count == step_idx:
                            log(f"\n검증 청크 찾음 (cot_chunks 인덱스: {chunk_idx})")
                            log(f"=== 원본 청크 ===")
                            log(chunk)
                            log(f"=== 원본 청크 끝 ===")
                            
                            updated_cot_chunks[chunk_idx] = chunk + f"<confidence>{confidence:.2f}</confidence>"
                            
                            log(f"\n=== 업데이트된 청크 ===")
                            log(updated_cot_chunks[chunk_idx])
                            log(f"=== 업데이트된 청크 끝 ===")
                            
                            chunk_found = True
                            break
                        verification_count += 1
                
                if not chunk_found:
                    log(f"\n⚠️ 검증 청크를 찾을 수 없음!")
                
                total_steps_processed += 1
                log(f"\nStep {step_idx + 1} 처리 완료!\n")
            
            # 업데이트된 데이터 저장
            updated_item = item.copy()
            updated_item['cot_chunks'] = updated_cot_chunks
            # cot도 업데이트 (모든 청크 합치기)
            updated_item['cot'] = ''.join(updated_cot_chunks)
            result_data.append(updated_item)
            
            log(f"\n{'=' * 70}")
            log(f"문제 {problem_idx} 처리 완료!")
            log(f"{'=' * 70}\n")
            
        except Exception as e:
            log(f"\n⚠️  문제 {problem_idx} 처리 중 오류 발생: {e}")
            log(traceback.format_exc())
            # 오류 발생 시 원본 데이터 그대로 추가
            result_data.append(item)
            n_skipped += 1
            continue
    
    # 5. 결과 저장 (섹션 번호 변경 [4/5])
    log(f"\n[4/5] 결과 저장 중...")
    
    if not result_data:
        log("    ❌ 생성된 데이터가 없습니다.")
        log_file.close()
        return 1
    
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        log(f"    ✓ 저장 완료: {OUTPUT_FILENAME}")
        log("\n" + "=" * 70)
        log("생성 완료!")
        log("=" * 70)
        log(f"처리된 문제 수:     {len(result_data):,}개")
        log(f"건너뛴 문제 수:     {n_skipped:,}개")
        log(f"처리된 스텝 수:     {total_steps_processed:,}개")
        log("=" * 70)
        log(f"\n디버그 로그: {DEBUG_LOG_FILENAME}")
        
        log_file.close()
        return 0
        
    except Exception as e:
        log(f"    ❌ 저장 실패: {e}")
        log(traceback.format_exc())
        log_file.close()
        return 1

if __name__ == "__main__":
    exit_code = main()
    
    print("정리 중...")
    os._exit(exit_code)