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
import openai
from tqdm import tqdm
import traceback

# ============================================================================
# 설정
# ============================================================================

# OpenAI 설정
OPENAI_API_KEY = ""
OPENAI_MODEL_NAME = "gpt-4o"  # gpt5-nano에서 gpt-4o로 변경

N_SAMPLES_PER_STEP = 10  # 각 스텝당 생성할 검증 CoT 개수
MAX_GENERATION_TOKENS = 2048
INPUT_FILENAME = "thinkprm_data.json"
OUTPUT_FILENAME = "thinkprm_data_conf.json"
DEBUG_LOG_FILENAME = "add_conf_debug.log"

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

def get_cot_prefix_before_step(cot_chunks, step_index):
    """
    step_index번째 검증 청크 직전까지의 모든 내용을 반환
    (모델이 이미 생성한 것처럼 인식하도록)
    
    Args:
        cot_chunks: 전체 cot_chunks 리스트
        step_index: 검증하려는 스텝의 인덱스 (0-based)
    
    Returns:
        step_index 직전까지의 모든 청크를 연결한 문자열
    """
    prefix_chunks = []
    verification_count = 0
    
    for chunk in cot_chunks:
        if is_verification_chunk(chunk):
            if verification_count == step_index:
                # 목표 검증 청크에 도달하면 중단
                break
            verification_count += 1
        prefix_chunks.append(chunk)
    
    return ''.join(prefix_chunks)

def extract_step_verification(text, step_number):
    """
    생성된 텍스트에서 특정 스텝의 검증 부분만 추출
    
    Args:
        text: 생성된 전체 텍스트
        step_number: 추출할 스텝 번호 (1-based)
    
    Returns:
        해당 스텝의 검증 텍스트와 라벨 (1: correct, 0: incorrect, None: 파싱 실패)
    """
    # Step N: 패턴 찾기
    pattern = rf'Step {step_number}:.*?\\boxed\{{(correct|incorrect)\}}'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        full_text = match.group(0)
        label = match.group(1).lower()
        return full_text, (1 if label == "correct" else 0)
    
    # 만약 Step N: 형식이 없다면 처음 나오는 boxed만 찾기
    pattern = r'\\boxed\{(correct|incorrect)\}'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        label = match.group(1).lower()
        return text, (1 if label == "correct" else 0)
    
    return None, None

def create_stop_sequence_after_boxed(text):
    """
    \\boxed{correct} 또는 \\boxed{incorrect} 다음에서 자를 수 있도록
    텍스트를 처리
    """
    pattern = r'(\\boxed\{(?:correct|incorrect)\})'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        end_pos = match.end()
        return text[:end_pos]
    return text

def format_verification_prompt(problem, prefix, cot_prefix):
    """
    OpenAI API에 맞는 메시지 리스트 생성
    원래 모델이 CoT를 생성했던 환경과 동일하게 설정
    
    Args:
        problem: 수학 문제
        prefix: 전체 풀이 과정 (원본 prefix 문자열)
        cot_prefix: 이미 생성된 것으로 간주할 이전 검증 내용
    """
    
    # System 메시지
    system_content = "You are a mathematical reasoning verification assistant."
    
    # User 메시지: 원래 모델이 받았던 것과 동일한 형식
    user_content = f"""Problem:
{problem}

Solution:
{prefix}

Please verify each step of the solution. For each step, provide a brief critique and conclude with exactly one of:
- The step is \\boxed{{correct}} (if the step is correct)
- The step is \\boxed{{incorrect}} (if the step contains an error)

Format your verification as:
Step 1: [Your critique for step 1]... The step is \\boxed{{correct/incorrect}}
Step 2: [Your critique for step 2]... The step is \\boxed{{correct/incorrect}}
...and so on for all steps."""

    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    
    # 이전 검증 내용이 있으면 assistant의 partial response로 추가
    if cot_prefix and cot_prefix != "<think>\n":
        messages.append({
            "role": "assistant",
            "content": cot_prefix  # 모델이 이미 생성한 것처럼 설정
        })
    else:
        # 첫 번째 스텝 검증의 경우 <think> 태그로 시작
        messages.append({
            "role": "assistant",
            "content": "<think>\n"
        })
    
    return messages

def print_full_prompt(messages, step_num):
    """
    전체 프롬프트 내용을 출력 (생략 없이)
    
    Args:
        messages: OpenAI API 메시지 리스트
        step_num: 현재 스텝 번호
    """
    log(f"\n{'='*80}")
    log(f"PROMPT DETAILS - Step {step_num}")
    log(f"{'='*80}")
    
    for i, msg in enumerate(messages):
        log(f"\n[{i+1}] Role: {msg['role']}")
        log(f"{'~'*40}")
        # 전체 내용을 생략 없이 출력
        log(msg['content'])
    
    log(f"\n{'='*80}\n")

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

    # 2. 입력 데이터 로드
    log(f"\n[2/5] 입력 데이터 로드 중...")
    log(f"    파일: {INPUT_FILENAME}")
    
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data = data[:2]  # 테스트용 제한
        
        log(f"    ✓ 로드 성공: {len(data):,}개 문제")
        
        total_steps_to_verify = sum(item['valid_prefix_step_count'] for item in data)
        total_api_calls = total_steps_to_verify * N_SAMPLES_PER_STEP
        log(f"    - 검증할 총 스텝 수: {total_steps_to_verify:,}개")
        log(f"    - 예상 API 호출 횟수: {total_api_calls:,}회 (스텝당 {N_SAMPLES_PER_STEP}회)")
        
    except Exception as e:
        log(f"    ❌ 로드 실패: {e}")
        log_file.close()
        return 1

    # 3. Confidence Score 생성
    log(f"\n[3/5] Confidence Score 생성 중...")
    log(f"    - 문제 수: {len(data)}")
    log(f"    - 스텝당 샘플링 횟수: {N_SAMPLES_PER_STEP}")
    log(f"    - Temperature: 1.4")
    log(f"    - Top-p: 0.9\n")
    
    result_data = []
    total_steps_processed = 0
    n_skipped = 0
    
    for problem_idx, item in enumerate(tqdm(data, desc="문제 처리 중")):
        try:
            problem = item['problem']
            prefix = item['prefix']  # prefix_steps 대신 원본 prefix 사용
            cot_chunks = item['cot_chunks']
            gt_step_labels = item['gt_step_labels']
            valid_prefix_step_count = item['valid_prefix_step_count']
            
            log(f"\n{'=' * 70}")
            log(f"문제 {problem_idx}")
            log(f"{'=' * 70}")
            log(f"문제 텍스트:\n{problem}")  # 전체 출력
            log(f"\nvalid_prefix_step_count: {valid_prefix_step_count}")
            log(f"gt_step_labels: {gt_step_labels}")
            
            # 업데이트할 cot_chunks 복사
            updated_cot_chunks = cot_chunks.copy()
            
            # valid_prefix_step_count만큼만 처리
            for step_idx in range(valid_prefix_step_count):
                log(f"\n{'-' * 70}")
                log(f"Step {step_idx + 1} 처리 시작")
                log(f"{'-' * 70}")
                
                # 해당 스텝 직전까지의 CoT 프리픽스 생성
                cot_prefix = get_cot_prefix_before_step(cot_chunks, step_idx)
                
                log(f"\nCoT Prefix 길이: {len(cot_prefix)} 문자")
                log(f"CoT Prefix 내용:")
                log(f"{cot_prefix}")  # 전체 출력
                
                # 프롬프트 생성 (원본 prefix 사용)
                messages = format_verification_prompt(
                    problem=problem,
                    prefix=prefix,  # 원본 prefix 그대로
                    cot_prefix=cot_prefix
                )
                
                log(f"\nAPI 요청 메시지 생성 완료\n")
                
                # 스텝별로 한 번만 프롬프트 출력
                current_step_number = step_idx + 1
                print_full_prompt(messages, current_step_number)
                
                # N_SAMPLES_PER_STEP번 생성
                log(f"API 요청 시작 (총 {N_SAMPLES_PER_STEP}번)...\n")
                
                generated_verifications = []
                
                for i in range(N_SAMPLES_PER_STEP):
                    log(f"    -> 샘플 {i + 1}/{N_SAMPLES_PER_STEP} 요청 중...")
                    
                    try:
                        # API 호출
                        response = client.chat.completions.create(
                            model=OPENAI_MODEL_NAME,
                            messages=messages,
                            max_completion_tokens=MAX_GENERATION_TOKENS,
                            temperature=1.4,  # 1.0에서 1.4로 변경
                            top_p=0.9,  # top_p 파라미터 추가
                            n=1
                        )
                        
                        # 생성된 결과
                        generated_text = response.choices[0].message.content
                        
                        # \\boxed{} 이후 텍스트 제거 (stop pattern 효과)
                        generated_text = create_stop_sequence_after_boxed(generated_text)
                        
                        log(f"\n    [샘플 {i + 1} 생성 결과 (전체)]")
                        log(f"    {generated_text}")  # 전체 출력
                        
                        generated_verifications.append(generated_text)
                    
                    except Exception as api_err:
                        log(f"    ❌ 샘플 {i + 1} 생성 중 오류 발생: {api_err}")
                        continue 
                
                log(f"\n총 {len(generated_verifications)}/{N_SAMPLES_PER_STEP}개 샘플 수집 완료.")
                
                # GT 라벨과 비교
                gt_label = gt_step_labels[step_idx]
                gt_numeric = 1 if gt_label == '+' else 0
                
                log(f"GT 라벨: {gt_label} ({gt_numeric})\n")
                
                # 각 생성 결과에서 현재 스텝의 검증 부분만 추출
                predicted_labels = []
                for i, gen_text in enumerate(generated_verifications):
                    # 현재 스텝의 검증 부분만 추출
                    step_verification, pred_label = extract_step_verification(gen_text, current_step_number)
                    
                    if step_verification:
                        log(f"\n[샘플 {i+1}] 추출된 Step {current_step_number} 검증:")
                        log(f"{step_verification}")  # 전체 출력
                    
                    predicted_labels.append(pred_label)
                    log(f"추출된 라벨: {pred_label} ({'correct' if pred_label == 1 else 'incorrect' if pred_label == 0 else 'PARSE_FAILED'})")
                
                # Confidence 계산
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
                            log(f"원본 청크:")
                            log(f"{chunk}")  # 전체 출력
                            updated_cot_chunks[chunk_idx] = chunk + f"<confidence>{confidence:.2f}</confidence>"
                            log(f"\n업데이트된 청크:")
                            log(f"{updated_cot_chunks[chunk_idx]}")  # 전체 출력
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
            result_data.append(item)
            n_skipped += 1
            continue
    
    # 4. 결과 저장
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