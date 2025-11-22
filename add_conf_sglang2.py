"""
ThinkPRM Confidence Score Generation Script (SGLang Ver.)

get_prefix_before_step 로직은 09:55:04 버전을 유지하고,
SGLang 호출 방식만 s.fork(n) + run_batch로 변경한 버전입니다.
"""

import json
import os
import sys
import re
from transformers import AutoTokenizer
from sglang import function, gen, set_default_backend, RuntimeEndpoint
from tqdm import tqdm
import traceback

# ============================================================================
# 설정
# ============================================================================

# SGLang 서버 설정
SGLANG_ENDPOINT = "http://127.0.0.1:31111"
MODEL_NAME_OR_PATH = "KirillR/QwQ-32B-Preview-AWQ"

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
# SGLang 생성 함수 (⭐️ s.fork(n) 방식으로 수정)
# ============================================================================

# ⭐️ [추가] s.fork(n) 방식은 결과를 저장할 전역 딕셔너리가 필요합니다.
prompt_to_states = {}

@function
def generate_step_verification(s, prompt: str, n: int):
    """
    SGLang 함수: s.fork(n)을 사용하여 N개 샘플링
    (generate_confidence_samples -> generate_step_verification 이름 변경)
    """
    s += prompt
    forks = s.fork(n) # ⭐️ gen(n=...) 대신 s.fork(n) 사용
    
    for fork in forks:
        fork += gen(
            "verification_step",  # 변수명은 이전과 동일하게 "verification_step" 사용
            max_tokens=MAX_GENERATION_TOKENS,
            temperature=1.0,
            #top_p=0.9
            # ⭐️ 요청하신 대로 stop_patterns는 적용하지 않음
        )
        
        # ⭐️ [추가] 결과를 전역 딕셔너리에 저장
        if prompt not in prompt_to_states:
            prompt_to_states[prompt] = []
        prompt_to_states[prompt].append(fork)

# ============================================================================
# 유틸리티 함수 (09:55:04 버전과 동일)
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
    """
    prefix_chunks = []
    verification_count = 0
    
    for chunk in cot_chunks:
        if is_verification_chunk(chunk):
            if verification_count == step_index:
                break
            verification_count += 1
        prefix_chunks.append(chunk)
    
    return ''.join(prefix_chunks)

def extract_step_verification(text, step_number):
    """
    생성된 텍스트에서 특정 스텝의 검증 부분만 추출
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

def format_verification_prompt(problem, prefix):
    """
    SGLang에서 템플릿 적용을 위해 사용할 [system, user] 메시지 리스트 생성
    """
    
    # System 메시지
    system_content = "You are a mathematical reasoning verification assistant."
    
    # User 메시지
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
    
    return messages

def print_full_prompt(messages, step_num):
    """
    전체 프롬프트 내용(messages 리스트)을 출력
    """
    log(f"\n{'='*80}")
    log(f"PROMPT DETAILS (Messages) - Step {step_num}")
    log(f"{'='*80}")
    
    for i, msg in enumerate(messages):
        log(f"\n[{i+1}] Role: {msg['role']}")
        log(f"{'~'*40}")
        log(msg['content'])
    
    log(f"\n{'='*80}\n")

# ============================================================================
# 메인 함수
# ============================================================================

def main():
    global log_file, prompt_to_states # ⭐️ 전역 변수 선언 추가
    
    # 로그 파일 열기
    log_file = open(DEBUG_LOG_FILENAME, 'w', encoding='utf-8')
    
    log("=" * 70)
    log(f"ThinkPRM Confidence Score Generation (SGLang Ver.)")
    log("=" * 70)
    
    # 1. SGLang 서버 연결
    log(f"\n[1/5] SGLang 서버 연결 중...")
    log(f"       Endpoint: {SGLANG_ENDPOINT}")
    
    try:
        set_default_backend(RuntimeEndpoint(SGLANG_ENDPOINT))
        log("       ✓ 연결 성공")
    except Exception as e:
        log(f"       ❌ 연결 실패: {e}")
        log_file.close()
        return 1

    # 2. 토크나이저 로드 (프롬프트 템플릿 적용에 필요)
    log(f"\n[2/5] 토크나이저 로드 중...")
    log(f"       Model: {MODEL_NAME_OR_PATH}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        log("       ✓ 로드 성공")
    except Exception as e:
        log(f"       ❌ 로드 실패: {e}")
        log_file.close()
        return 1

    # 3. 입력 데이터 로드
    log(f"\n[3/5] 입력 데이터 로드 중...")
    log(f"       파일: {INPUT_FILENAME}")
    
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data = data[:2]
        
        log(f"       ✓ 로드 성공: {len(data):,}개 문제")
        
        total_steps_to_verify = sum(item['valid_prefix_step_count'] for item in data)
        total_api_calls = total_steps_to_verify
        log(f"       - 검증할 총 스텝 수: {total_steps_to_verify:,}개")
        log(f"       - SGLang 호출 횟수: {total_api_calls:,}회 (호출당 {N_SAMPLES_PER_STEP}개 샘플링)")
        
    except Exception as e:
        log(f"       ❌ 로드 실패: {e}")
        log_file.close()
        return 1

    # 4. Confidence Score 생성
    log(f"\n[4/5] Confidence Score 생성 중...")
    log(f"       - 문제 수: {len(data)}")
    log(f"       - 스텝당 샘플링 횟수: {N_SAMPLES_PER_STEP}")
    log(f"       - Temperature: 1.4")
    log(f"       - Top-p: 0.9\n")
    
    result_data = []
    total_steps_processed = 0
    n_skipped = 0
    
    for problem_idx, item in enumerate(tqdm(data, desc="문제 처리 중")):
        try:
            problem = item['problem']
            prefix = item['prefix']
            cot_chunks = item['cot_chunks']
            gt_step_labels = item['gt_step_labels']
            valid_prefix_step_count = item['valid_prefix_step_count']
            
            log(f"\n{'=' * 70}")
            log(f"문제 {problem_idx}")
            log(f"{'=' * 70}")
            log(f"문제 텍스트:\n{problem}")
            log(f"\nvalid_prefix_step_count: {valid_prefix_step_count}")
            log(f"gt_step_labels: {gt_step_labels}")
            
            updated_cot_chunks = cot_chunks.copy()
            
            for step_idx in range(valid_prefix_step_count):
                log(f"\n{'-' * 70}")
                log(f"Step {step_idx + 1} 처리 시작")
                log(f"{'-' * 70}")
                
                # ⭐️ [유지] 09:55:04 버전의 cot_prefix 생성 로직
                cot_prefix = get_cot_prefix_before_step(cot_chunks, step_idx)
                
                log(f"\nCoT Prefix 길이: {len(cot_prefix)} 문자")
                log(f"CoT Prefix 내용:")
                log(f"{cot_prefix}")
                
                # ⭐️ [유지] 09:39:13 버전의 프롬프트 생성 로직
                messages = format_verification_prompt(
                    problem=problem,
                    prefix=prefix
                )
                
                current_step_number = step_idx + 1
                
                print_full_prompt(messages, current_step_number)

                try:
                    prompt_string_base = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True 
                    )
                except Exception as e:
                    log(f"       ❌ 챗 템플릿 적용 실패: {e}")
                    prompt_string_base = f"{messages[1]['content']}\n\n"

                final_prompt_string = prompt_string_base + cot_prefix


                log("\n--- SGLang Final Prompt String (tokens applied) ---")
                log(final_prompt_string) 
                log("--- End of SGLang Prompt ---\n")

                # ⭐️ [수정됨] SGLang API 호출 방식을 run_batch + s.fork(n)로 변경
                log(f"SGLang API 요청 시작 (n={N_SAMPLES_PER_STEP} 샘플링, run_batch)...\n")
                
                # ⭐️ [추가됨] 전역 변수 초기화
                prompt_to_states = {}
                
                generated_verifications = []
                try:
                    # ⭐️ [수정됨] run_batch 호출
                    batch_args = [{'prompt': final_prompt_string, 'n': N_SAMPLES_PER_STEP}]
                    _ = generate_step_verification.run_batch(batch_args) # ⭐️ 새 SGLang 함수
                    
                    # ⭐️ [수정됨] global dict에서 결과 수집
                    if final_prompt_string not in prompt_to_states:
                        # 09:55:04 버전의 오류와 동일한 상황 (KeyError)
                        raise KeyError(f"'verification_step'. SGLang이 프롬프트에 대해 생성을 못했습니다.")

                    for state in prompt_to_states[final_prompt_string]:
                        # ⭐️ 변수명 "verification_step" 유지
                        verification = state["verification_step"] 
                        generated_verifications.append(verification.strip())

                    log(f"       ✓ SGLang 생성 완료. {len(generated_verifications)}개 샘플 수신.")

                except Exception as api_err:
                    log(f"       ❌ SGLang API 호출 중 오류 발생: {api_err}")
                    log(traceback.format_exc())
                    continue
                
                # [이하 코드는 09:39:13 버전과 동일]
                
                for i, gen_text in enumerate(generated_verifications):
                    cleaned_text = create_stop_sequence_after_boxed(gen_text)
                    
                    log(f"\n       [샘플 {i + 1} 생성 결과 (정리됨)]")
                    log(f"       {cleaned_text}")
                    
                    generated_verifications[i] = cleaned_text

                
                log(f"\n총 {len(generated_verifications)}/{N_SAMPLES_PER_STEP}개 샘플 수집 완료.")
                
                gt_label = gt_step_labels[step_idx]
                gt_numeric = 1 if gt_label == '+' else 0
                
                log(f"GT 라벨: {gt_label} ({gt_numeric})\n")
                
                predicted_labels = []
                for i, gen_text in enumerate(generated_verifications):
                    step_verification, pred_label = extract_step_verification(gen_text, current_step_number)
                    
                    if step_verification:
                        log(f"\n[샘플 {i+1}] 추출된 Step {current_step_number} 검증:")
                        log(f"{step_verification}")
                    
                    predicted_labels.append(pred_label)
                    log(f"추출된 라벨: {pred_label} ({'correct' if pred_label == 1 else 'incorrect' if pred_label == 0 else 'PARSE_FAILED'})")
                
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
                
                verification_count = 0
                chunk_found = False
                for chunk_idx, chunk in enumerate(updated_cot_chunks):
                    if is_verification_chunk(chunk):
                        if verification_count == step_idx:
                            log(f"\n검증 청크 찾음 (cot_chunks 인덱스: {chunk_idx})")
                            log(f"원본 청크:\n{chunk}")
                            updated_cot_chunks[chunk_idx] = chunk + f"<confidence>{confidence:.2f}</confidence>"
                            log(f"\n업데이트된 청크:\n{updated_cot_chunks[chunk_idx]}")
                            chunk_found = True
                            break
                        verification_count += 1
                
                if not chunk_found:
                    log(f"\n⚠️ 검증 청크를 찾을 수 없음!")
                
                total_steps_processed += 1
                log(f"\nStep {step_idx + 1} 처리 완료!\n")
            
            updated_item = item.copy()
            updated_item['cot_chunks'] = updated_cot_chunks
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
    
    # 5. 결과 저장
    log(f"\n[5/5] 결과 저장 중...")
    
    if not result_data:
        log("       ❌ 생성된 데이터가 없습니다.")
        log_file.close()
        return 1
    
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        log(f"       ✓ 저장 완료: {OUTPUT_FILENAME}")
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
        log(f"       ❌ 저장 실패: {e}")
        log(traceback.format_exc())
        log_file.close()
        return 1

if __name__ == "__main__":
    exit_code = main()
    
    print("정리 중...")
    os._exit(exit_code)