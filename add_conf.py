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
   conda activate thinkprm
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
    
    for chunk in cot_chunks[1:]:
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
    SGLang에 맞는 프롬프트 생성 (항상 9개의 Few-shot 예시 포함)
    """
    
    # 기본 사용자 컨텐츠
    user_content = f"""Problem:
{problem}

Solution:
{prefix}

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE EXACTLY:

You are a mathematical verification assistant. You MUST verify each step of the solution above.

FOR EACH STEP, YOU MUST:
1. Start with "Step N:" (e.g., "Step 1:", "Step 2:").
2. Provide a brief mathematical critique.
3. End with EXACTLY one of these two phrases:
   - Thus, the step is \\boxed{{correct}}. The evaluation for this step ends here.
   - Thus, the step is \\boxed{{incorrect}}. The evaluation for this step ends here.

IMPORTANT RULES:
- **ALL STEPS**: You MUST verify EVERY step, including the final "# Answer" step.
- **STEP 1**: Do NOT use markdown headers like "**Step 1:**". Just write "Step 1:" as plain text.
- **ANSWER STEP**: If the step involves "# Answer", check if the final value is mathematically consistent with previous steps. Do NOT output "N/A", "Conclusion", or "[Not a solution step]". Treat it exactly like any other step.
- **FORMAT**: Do NOT use square brackets like [boxed{{correct}}]. You MUST use LaTeX format: \\boxed{{correct}} or \\boxed{{incorrect}}.
- **ENDING**: You MUST use the exact closing phrase provided above. Do NOT just say "The answer is correct" or "The step is valid".
"""

    # -------------------------------------------------------------------------
    # Few-shot Examples (총 9개: 실패 사례 5개 + 성공 사례 4개)
    # -------------------------------------------------------------------------
    few_shot_examples = """
=== INCORRECT EXAMPLES (DO NOT DO THIS) ===

[Bad Example 1: Last step using value in box instead of correct/incorrect]
Problem: Calculate 3 + 4.
Solution:
Step 5: # Answer 7
Your answer:
Step 5: # Answer 7 Critique: The calculation 3+4=7 is correct. Thus, the step is \\boxed{7}. The evaluation for this step ends here.
(ERROR: The box must contain 'correct' or 'incorrect', not the number 7.)

[Bad Example 2: Last step missing the mandatory closing phrase]
Problem: Multiply 3 by 5.
Solution:
Step 4: # Answer 15
Your answer:
Step 4: # Answer 15 Critique: The final answer matches the derivation. The solution is correct.
(ERROR: Missing the mandatory phrase "Thus, the step is \\boxed{correct}. The evaluation for this step ends here.")

[Bad Example 3: Step 1 format violation]
Problem: Find the perimeter of a rectangle with width 5.
Solution:
Step 1: Let w be the width.
Your answer:
This step is correct because defining variables is a good start.
(ERROR: Must start with "Step 1:" and follow the full format.)

[Bad Example 4: Refusing to verify the Answer step]
Problem: What is 6 times 7?
Solution:
Step 6: # Answer 42
Your answer:
Step 6: # Answer 42 Critique: This is just the final answer format. [Not a solution step]
(ERROR: You must verify the answer step just like any other step.)

[Bad Example 5: Using N/A in the box]
Problem: Evaluate 2 + 5.
Solution:
Step 3: # Answer 7
Your answer:
Step 3: # Answer 7 Critique: There is no calculation here. Conclusion: \\boxed{N/A}
(ERROR: Never use N/A. If the number is correct based on previous steps, it is \\boxed{correct}.)

=== CORRECT EXAMPLES (FOLLOW THIS PATTERN) ===

[Good Example 1: Standard Step Verification]
Problem: Solve 3x = 12
Solution:
Step 1: Divide both sides by 3: x = 4.
Your answer:
Step 1: Divide both sides by 3: x = 4. Critique: Dividing both sides of 3x = 12 by 3 yields x = 4. The arithmetic is correct. Thus, the step is \\boxed{correct}. The evaluation for this step ends here.

[Good Example 2: Step 1 Verification (Clean Start)]
Problem: Find the area of a circle with radius 2.
Solution:
Step 1: The formula for area is $\\pi r^2$.
Your answer:
Step 1: The formula for area is $\\pi r^2$. Critique: The area of a circle is calculated using the formula $A = \\pi r^2$. This is the correct formula. Thus, the step is \\boxed{correct}. The evaluation for this step ends here.

[Good Example 3: Last Step (# Answer) Verification]
Problem: Calculate 10 - 2.
Solution:
Step 2: # Answer 8
Your answer:
Step 2: # Answer 8 Critique: The previous steps lead to 10 - 2. Since 10 - 2 = 8, the final answer provided is correct. Thus, the step is \\boxed{correct}. The evaluation for this step ends here.

[Good Example 4: Step 1 with existing context (Continuation)]
(Context provided: <think>Okay, let's verify step 1.\n)
Problem: Simplify 2a + 3a.
Solution:
Step 1: 2a + 3a = 5a.
Your answer:
Step 1: 2a + 3a = 5a. Critique: Combining like terms 2a and 3a results in 5a. This is algebraically correct. Thus, the step is \\boxed{correct}. The evaluation for this step ends here.
"""

    # 프롬프트 조합 (항상 Few-shot 예시 포함)
    full_prompt = user_content + "\n\n" + few_shot_examples + "\n\n" + "Your answer:" + "\n\n" + cot_prefix
    
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
    
    stop_patterns = [
        "The evaluation for this step ends here.",
    ]
    
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
            stop = stop_patterns,
            #top_p=1.0,
            #regex=r"Step\s+\d+:\s.*?Critique:\s.*?The step is \\boxed\{(correct|incorrect)\}"
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
        
        data = data[:]  # 테스트용 제한
        
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