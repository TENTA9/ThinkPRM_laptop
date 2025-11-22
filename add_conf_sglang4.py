"""
ThinkPRM Confidence Score Generation Script (SGLang Ver.)

thinkprm_data.json의 각 검증 스텝에 confidence score를 추가하여
thinkprm_data_conf.json을 생성합니다.

사용법:
1. 터미널 1에서 SGLang 서버 실행:
   conda activate thinkprm
   export LD_LIBRARY_PATH=~/fake_cuda/lib64:$LD_LIBRARY_PATH
   export LIBRARY_PATH=~/fake_cuda/lib64:$LIBRARY_PATH
   python -m sglang.launch_server --model-path "KirillR/QwQ-32B-Preview-AWQ" --port 31111 --host 127.0.0.1 --disable-radix-cache

2. 터미널 2에서 이 스크립트 실행:
   python add_conf.py
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
SGLANG_ENDPOINT = "http://127.0.0.1:31111"
MODEL_NAME_OR_PATH = "KirillR/QwQ-32B-Preview-AWQ"
N_SAMPLES_PER_STEP = 10  
MAX_GENERATION_TOKENS = 4096
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

def format_verification_prompt(problem, prefix, step_idx, cot_prefix):
    """
    SGLang에 맞는 프롬프트 생성
    
    Args:
        problem: 수학 문제
        prefix: 전체 풀이 과정 (원본 prefix 문자열)
        step_idx: 현재 검증 스텝 인덱스 (0-based)
        cot_prefix: 이미 생성된 것으로 간주할 이전 검증 내용
    """
    
    # 기본 사용자 컨텐츠
    user_content = f"""Problem:
{problem}

Solution:
{prefix}

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE EXACTLY:

You are a mathematical verification assistant. You MUST verify each step of the solution above.

FOR EACH STEP, YOU MUST:
1. Write "Step N:" where N is the step number
2. Provide a brief mathematical critique explaining why the step is correct or incorrect
3. End with EXACTLY one of these two phrases:
   - "The step is \\boxed{{correct}}" (if the step is mathematically valid)
   - "The step is \\boxed{{incorrect}}" (if the step contains an error)

AFTER PROVIDING YOUR CRITIQUE AND CONCLUSION FOR ALL THE STEPS, YOU MUST:
1. summarize whether the entire solution is correct or not with a simple "Yes" or "No"
  - "Is the solution correct? Yes" (If all steps are correct)
  - "Is the solution correct? No" (If any step is incorrect) 

MANDATORY FORMAT:
Step 1: [The full original content of Step 1] Critique: [Your mathematical analysis of step 1]. The step is \\boxed{{correct}}
Step 2: [The full original content of Step 2] Critique: [Your mathematical analysis of step 2]. The step is \\boxed{{incorrect}}
Step 3: [The full original content of Step 3] Critique: [Your mathematical analysis of step 3]. The step is \\boxed{{correct}}
(Continue for ALL steps in the solution)
Is the solution correct? Yes/No

IMPORTANT RULES:
- You MUST verify EVERY step in the solution above
- You MUST use the EXACT format shown above
- You MUST end each step verification with "The step is \\boxed{{correct}}" or "The step is \\boxed{{incorrect}}"
- DO NOT use any other format or wording
- DO NOT skip any steps
- DO NOT add extra commentary outside the step verifications"""

    # Few-shot 예시
    few_shot_example = """
EXAMPLE:
Problem: Solve x + 2 = 5
Solution:
Step 1: Subtract 2 from both sides: x = 5 - 2
Step 2: Therefore x = 4

Answer(example):
<think>
Step 1: Subtract 2 from both sides: x = 5 - 2 Critique: Subtracting 2 from both sides is the correct algebraic operation to isolate x. The step is \\boxed{correct}
Step 2: Therefore x = 4 Critique: The arithmetic 5 - 2 = 4 is incorrect. The correct result should be 3. The step is \\boxed{incorrect}
</think>
Is the solution correct? No
"""

    # 프롬프트 조합
    if step_idx == 0:
        # 첫 번째 스텝: few-shot 예시 포함
        full_prompt = user_content + "\n\n" + few_shot_example + "\n\n" + "Your answer:" + "\n\n" + cot_prefix
    else:
        # 이후 스텝: few-shot 없이
        full_prompt = user_content + "\n\n" + "Your answer:" + "\n\n" + cot_prefix
    
    return full_prompt

def print_full_prompt(prompt, step_num):
    """
    전체 프롬프트 내용을 출력
    
    Args:
        prompt: 프롬프트 문자열
        step_num: 현재 스텝 번호
    """
    log(f"\n{'='*80}")
    log(f"PROMPT DETAILS - Step {step_num}")
    log(f"{'='*80}")
    log(prompt)
    log(f"\n{'='*80}\n")

# ============================================================================
# SGLang 생성 함수
# ============================================================================

prompt_to_states = {}

# ⭐️ [수정됨] .run_batch()를 사용하도록 함수 로직 변경
# .run()처럼 state를 반환하는 대신, .run_batch()가 global dict에 저장하도록 함
@function
def generate_step_verification(s, prompt: str, num_samples: int):
    """
    SGLang를 사용하여 n개의 검증 생성 (stop 없이)
    .run_batch()용으로 수정됨:
    - forks를 반환하는 대신, global dict 'prompt_to_states'에 state를 저장
    """
    s += prompt
    
    forks = s.fork(num_samples)
    
    # .run_batch() 스타일:
    # 1. 단일 이름으로 gen()을 호출
    # 2. state를 global dict에 저장
    for fork in forks:
        fork += gen(
            "verification_output",  # ⭐️ 단일 변수 이름 사용
            max_tokens=MAX_GENERATION_TOKENS,
            temperature=1.0,
            #top_p=0.9
        )
        
        # ⭐️ [추가] SGLang이 이 state를 global dict에 저장하도록 함
        # (이 코드는 SGLang 백엔드에서 실행됨)
        if prompt not in prompt_to_states:
            prompt_to_states[prompt] = []
        prompt_to_states[prompt].append(fork)
    
    # ⭐️ .run_batch()는 반환값이 필요 없음 (None 반환)

# ============================================================================
# 메인 함수
# ============================================================================

def main():
    global log_file
    
    # 로그 파일 열기
    log_file = open(DEBUG_LOG_FILENAME, 'w', encoding='utf-8')
    
    log("=" * 70)
    log("ThinkPRM Confidence Score Generation (SGLang Ver.)")
    log("=" * 70)
    
    # 1. SGLang 서버 연결
    log(f"\n[1/5] SGLang 서버 연결 중...")
    log(f"     Endpoint: {SGLANG_ENDPOINT}")
    
    try:
        set_default_backend(RuntimeEndpoint(SGLANG_ENDPOINT))
        log(f"     ✓ 연결 성공")
    except Exception as e:
        log(f"     ❌ 연결 실패: {e}")
        log(f"\n서버를 먼저 실행하세요:")
        log(f"conda activate thinkprm")
        log(f"export LD_LIBRARY_PATH=~/fake_cuda/lib64:$LD_LIBRARY_PATH")
        log(f"export LIBRARY_PATH=~/fake_cuda/lib64:$LIBRARY_PATH")
        log(f'python -m sglang.launch_server --model-path "{MODEL_NAME_OR_PATH}" --port 31111 --host 127.0.0.1 --disable-radix-cache')
        log_file.close()
        return 1

    # 2. 토크나이저 로드
    log(f"\n[2/5] 토크나이저 로드 중...")
    log(f"     Model: {MODEL_NAME_OR_PATH}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        log(f"     ✓ 로드 성공")
    except Exception as e:
        log(f"     ❌ 로드 실패: {e}")
        log_file.close()
        return 1

    # 3. 입력 데이터 로드
    log(f"\n[3/5] 입력 데이터 로드 중...")
    log(f"     파일: {INPUT_FILENAME}")
    
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data = data[:3]  # 테스트용 제한
        
        log(f"     ✓ 로드 성공: {len(data):,}개 문제")
        
        total_steps_to_verify = sum(item['valid_prefix_step_count'] for item in data)
        # ⭐️ .run_batch()를 사용하므로 API 호출 횟수가 스텝 수와 같음
        total_api_calls = total_steps_to_verify
        log(f"     - 검증할 총 스텝 수: {total_steps_to_verify:,}개")
        log(f"     - 예상 SGLang 호출 횟수: {total_api_calls:,}회 (호출당 {N_SAMPLES_PER_STEP}개 샘플)")
        
    except Exception as e:
        log(f"     ❌ 로드 실패: {e}")
        log_file.close()
        return 1

    # 4. Confidence Score 생성
    log(f"\n[4/5] Confidence Score 생성 중...")
    log(f"     - 문제 수: {len(data)}")
    log(f"     - 스텝당 샘플링 횟수: {N_SAMPLES_PER_STEP}")
    log(f"     - Temperature: 1.4")
    log(f"     - Top-p: 0.9\n")
    
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
                log(f"{cot_prefix}")
                
                # 프롬프트 생성
                prompt = format_verification_prompt(
                    problem=problem,
                    prefix=prefix,
                    step_idx=step_idx,
                    cot_prefix=cot_prefix
                )
                
                log(f"\n프롬프트 생성 완료\n")
                
                # 스텝별로 한 번만 프롬프트 출력
                current_step_number = step_idx + 1
                print_full_prompt(prompt, current_step_number)
                
                # prompt_to_states 초기화
                global prompt_to_states
                prompt_to_states = {}
                
                # N_SAMPLES_PER_STEP번 생성 (병렬)
                log(f"생성 시작 (총 {N_SAMPLES_PER_STEP}개 병렬 생성)...\n")
                
                generated_verifications = []
                try:
                    # ⭐️ [수정됨] .run() 대신 .run_batch() 사용
                    log("DEBUG: .run_batch() 호출 시도 중...")
                    batch_args = [{'prompt': prompt, 'num_samples': N_SAMPLES_PER_STEP}]
                    # .run_batch()는 비동기이며, SGLang 함수 내부에서 'prompt_to_states'에 저장함
                    _ = generate_step_verification.run_batch(batch_args)
                    log("DEBUG: .run_batch() 호출 성공.")

                    # ⭐️ [수정됨] global dict에서 결과 수집
                    if prompt not in prompt_to_states:
                        # SGLang 서버에서 작업은 했으나, 0개의 state가 반환된 경우
                        log(f"     ❌ SGLang이 프롬프트에 대해 생성을 못했습니다. (KeyError)")
                        raise KeyError(f"SGLang이 프롬프트에 대해 생성을 못했습니다.")

                    # ⭐️ [수정됨] state 리스트에서 결과 추출
                    states = prompt_to_states[prompt] # state 리스트
                    log(f"DEBUG: {len(states)}개 state 수신 완료.")

                    for i, state in enumerate(states):
                        # ⭐️ SGLang 함수에서 정의한 "verification_output" 사용
                        generated_text = state["verification_output"]
                        
                        # \boxed{} 이후 텍스트 제거
                        generated_text = create_stop_sequence_after_boxed(generated_text)
                        
                        log(f"\n     [샘플 {i + 1} 생성 결과]")
                        log(f"     {generated_text}")
                        
                        generated_verifications.append(generated_text)
                
                except Exception as gen_err:
                    log(f"     ❌ 생성 중 오류 발생: {gen_err}")
                    log(traceback.format_exc()) # <--- 상세 트레이스백 추가
                    log("DEBUG: .run_batch() 호출 중 예외 발생.")
                    generated_verifications = [] # 다음 단계로 넘어가지 않도록 비움
                
                log(f"\n총 {len(generated_verifications)}/{N_SAMPLES_PER_STEP}개 샘플 수집 완료.")
                
                # GT 라벨과 비교
                gt_label = gt_step_labels[step_idx]
                gt_numeric = 1 if gt_label == '+' else 0
                
                log(f"GT 라벨: {gt_label} ({gt_numeric})\n")
                
                # 각 생성 결과에서 현재 스텝의 검증 부분만 추출
                predicted_labels = []
                for i, gen_text in enumerate(generated_verifications):
                    step_verification, pred_label = extract_step_verification(gen_text, current_step_number)
                    
                    if step_verification:
                        log(f"\n[샘플 {i+1}] 추출된 Step {current_step_number} 검증:")
                        log(f"{step_verification}")
                    
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
                            log(f"{chunk}")
                            updated_cot_chunks[chunk_idx] = chunk + f"<confidence>{confidence:.2f}</confidence>"
                            log(f"\n업데이트된 청크:")
                            log(f"{updated_cot_chunks[chunk_idx]}")
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
        log("     ❌ 생성된 데이터가 없습니다.")
        log_file.close()
        return 1
    
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        log(f"     ✓ 저장 완료: {OUTPUT_FILENAME}")
        log("\n" + "=" * 70)
        log("생성 완료!")
        log("=" * 70)
        log(f"처리된 문제 수:      {len(result_data):,}개")
        log(f"건너뛴 문제 수:      {n_skipped:,}개")
        log(f"처리된 스텝 수:      {total_steps_processed:,}개")
        log("=" * 70)
        log(f"\n디버그 로그: {DEBUG_LOG_FILENAME}")
        
        log_file.close()
        return 0
        
    except Exception as e:
        log(f"     ❌ 저장 실패: {e}")
        log(traceback.format_exc())
        log_file.close()
        return 1

if __name__ == "__main__":
    exit_code = main()
    
    print("정리 중...")
    os._exit(exit_code)