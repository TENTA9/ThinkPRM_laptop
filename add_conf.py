"""
ThinkPRM Confidence Score Generation Script (Final Fixed Prompt Ver.)
- Terminal: Shows ONLY progress bar (tqdm)
- Log File: Records EVERYTHING (Detailed samples, prompt, parsing results)
- Prompt: RESTORED to original requested version (Exact Match)
"""

import json
import os
import sys
import re
import traceback
import argparse
from transformers import AutoTokenizer
from sglang import function, gen, set_default_backend, RuntimeEndpoint
from tqdm import tqdm

# ============================================================================
# [기본 설정값]
# ============================================================================
DEFAULT_SGLANG_ENDPOINT = "http://127.0.0.1:31111"
DEFAULT_MODEL_PATH = "KirillR/QwQ-32B-Preview-AWQ"
DEFAULT_N_SAMPLES = 10
DEFAULT_MAX_TOKENS = 4096
DEFAULT_INPUT_FILE = "thinkprm_data.json"
DEFAULT_OUTPUT_FILE = "thinkprm_data_conf.json"
DEFAULT_LOG_FILE = "add_conf_debug.log"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SAVE_INTERVAL = 10

# 전역 변수 (args로 덮어씌워짐)
SGLANG_ENDPOINT = DEFAULT_SGLANG_ENDPOINT
MODEL_NAME_OR_PATH = DEFAULT_MODEL_PATH
N_SAMPLES_PER_STEP = DEFAULT_N_SAMPLES
MAX_GENERATION_TOKENS = DEFAULT_MAX_TOKENS
TEMPERATURE = DEFAULT_TEMPERATURE
DEBUG_LOG_FILENAME = DEFAULT_LOG_FILE

# ============================================================================
# 로깅 함수
# ============================================================================
log_file = None

def log(message, console=False):
    """
    console=False: 파일에만 기록 (tqdm 진행바 보호)
    console=True: 파일+콘솔 둘 다 출력 (에러, 시작 메시지 등)
    """
    if console:
        print(message)
    
    if log_file:
        log_file.write(str(message) + "\n")
        log_file.flush()

# ============================================================================
# 유틸리티 함수
# ============================================================================
def is_verification_chunk(chunk):
    chunk = chunk.strip()
    if not chunk.startswith("Step"): return False
    if "\\boxed{" not in chunk: return False
    return True

def get_cot_prefix_before_step(cot_chunks, step_index):
    prefix_chunks = []
    verification_count = 0
    for chunk in cot_chunks[1:]:
        if is_verification_chunk(chunk):
            if verification_count == step_index: break
            verification_count += 1
        prefix_chunks.append(chunk)
    return ''.join(prefix_chunks)

def extract_step_verification(text, step_number):
    # Step N: ... \boxed{correct} 형식 찾기
    pattern = rf'Step {step_number}:.*?\\boxed\{{(correct|incorrect)\}}'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0), (1 if match.group(1).lower() == "correct" else 0)
    
    # 만약 형식이 깨져서 \boxed{correct}만 있는 경우
    pattern = r'\\boxed\{(correct|incorrect)\}'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return text, (1 if match.group(1).lower() == "correct" else 0)
    return None, None

def create_stop_sequence_after_boxed(text):
    pattern = r'(\\boxed\{(?:correct|incorrect)\})'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return text[:match.end()]
    return text

# ⭐️ [수정됨] 요청하신 프롬프트 원본 그대로 적용
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
    log(f"\n{'='*80}\nPROMPT DETAILS - Step {step_num}\n{'='*80}\n{prompt}\n{'='*80}\n")

# ============================================================================
# SGLang 생성 함수
# ============================================================================
prompt_to_states = {}

@function
def generate_step_verification(s, prompt: str, num_samples: int):
    stop_patterns = ["The evaluation for this step ends here."]
    s += prompt
    forks = s.fork(num_samples)
    for fork in forks:
        fork += gen(
            "verification_output",
            max_tokens=MAX_GENERATION_TOKENS,
            temperature=TEMPERATURE,
            stop=stop_patterns,
        )
        if prompt not in prompt_to_states:
            prompt_to_states[prompt] = []
        prompt_to_states[prompt].append(fork)

# ============================================================================
# Argument Parser
# ============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="ThinkPRM Confidence Generation")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_SGLANG_ENDPOINT)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--log-file", type=str, default=DEFAULT_LOG_FILE)
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    return parser.parse_args()

# ============================================================================
# 메인 함수
# ============================================================================
def main():
    global log_file, SGLANG_ENDPOINT, MODEL_NAME_OR_PATH, N_SAMPLES_PER_STEP
    global MAX_GENERATION_TOKENS, TEMPERATURE, DEBUG_LOG_FILENAME, OUTPUT_FILENAME

    args = parse_arguments()
    
    # 자동 파일명 생성
    if args.output == DEFAULT_OUTPUT_FILE and (args.start != 0 or args.end != -1):
        base_name, ext = os.path.splitext(DEFAULT_OUTPUT_FILE)
        end_str = args.end if args.end != -1 else "end"
        args.output = f"{base_name}_{args.start}_{end_str}{ext}"
    
    SGLANG_ENDPOINT = args.endpoint
    MODEL_NAME_OR_PATH = args.model_path
    N_SAMPLES_PER_STEP = args.n_samples
    MAX_GENERATION_TOKENS = args.max_tokens
    TEMPERATURE = args.temperature
    DEBUG_LOG_FILENAME = args.log_file
    OUTPUT_FILENAME = args.output
    
    log_file = open(DEBUG_LOG_FILENAME, 'a', encoding='utf-8')
    
    # 시작 정보는 콘솔에도 출력
    log("=" * 70, console=True)
    log(f"ThinkPRM Confidence Generation (Started)", console=True)
    log("=" * 70, console=True)
    log(f" - Range: {args.start} ~ {'EOF' if args.end == -1 else args.end}", console=True)
    log(f" - Output: {OUTPUT_FILENAME}", console=True)
    log(f" - Log File: {DEBUG_LOG_FILENAME} (Details here)", console=True)

    try:
        set_default_backend(RuntimeEndpoint(SGLANG_ENDPOINT))
        log("✓ SGLang 서버 연결 성공", console=True)
    except Exception as e:
        log(f"❌ 서버 연결 실패: {e}", console=True)
        return 1

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
            
        end_idx = args.end if args.end != -1 else len(full_data)
        result_data = []
        processed_count = 0
        
        if os.path.exists(OUTPUT_FILENAME):
            try:
                with open(OUTPUT_FILENAME, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                processed_count = len(result_data)
                log(f"🔄 기존 작업 파일 로드: {processed_count}개 완료됨", console=True)
            except:
                result_data = []

        real_start_idx = args.start + processed_count
        
        if real_start_idx >= end_idx:
            log("✅ 이미 완료된 작업입니다.", console=True)
            return 0
        
        target_data = full_data[real_start_idx : end_idx]
        log(f"🚀 작업 시작: {len(target_data)}개 문제 처리 중...\n", console=True)
        
    except Exception as e:
        log(f"❌ 데이터 준비 실패: {e}", console=True)
        return 1

    # 메인 루프
    for i, item in enumerate(tqdm(target_data, desc="Processing", initial=processed_count, total=end_idx - args.start)):
        current_idx = real_start_idx + i
        
        try:
            problem = item['problem']
            prefix = item['prefix']
            cot_chunks = item['cot_chunks']
            gt_step_labels = item['gt_step_labels']
            valid_prefix_step_count = item['valid_prefix_step_count']
            
            updated_cot_chunks = cot_chunks.copy()
            
            log(f"\n\n{'='*30} Problem {current_idx} {'='*30}")
            
            for step_idx in range(valid_prefix_step_count):
                cot_prefix = get_cot_prefix_before_step(cot_chunks, step_idx)
                prompt = format_verification_prompt(problem, prefix, step_idx, cot_prefix)
                current_step_number = step_idx + 1
                
                print_full_prompt(prompt, current_step_number) # 파일에만 기록됨
                
                global prompt_to_states
                prompt_to_states = {}
                batch_args = [{'prompt': prompt, 'num_samples': N_SAMPLES_PER_STEP}]
                
                try:
                    _ = generate_step_verification.run_batch(batch_args)
                    if prompt not in prompt_to_states: raise KeyError("No Output")
                    states = prompt_to_states[prompt]
                    generated_verifications = [create_stop_sequence_after_boxed(s["verification_output"]) for s in states]
                except Exception as gen_err:
                    log(f"❌ Generation Error: {gen_err}")
                    generated_verifications = []

                # 상세 파싱 결과 로그 기록
                gt_label = gt_step_labels[step_idx]
                gt_numeric = 1 if gt_label == '+' else 0
                
                predicted_labels = []
                
                log(f"\n--- Step {current_step_number} Generation Results ({len(generated_verifications)} samples) ---")
                
                for s_i, gen_text in enumerate(generated_verifications):
                    step_verification, pred_label = extract_step_verification(gen_text, current_step_number)
                    predicted_labels.append(pred_label)
                    
                    # 상세 내용을 파일에 기록
                    log(f"\n[Sample {s_i+1}]")
                    log(f"Generated: {gen_text.strip()}")
                    label_str = 'Correct' if pred_label == 1 else 'Incorrect' if pred_label == 0 else 'FAIL(None)'
                    log(f"Extracted: {label_str} ({pred_label})")

                matches = sum(1 for pred in predicted_labels if pred == gt_numeric)
                confidence = matches / N_SAMPLES_PER_STEP
                
                # 요약 정보 기록
                parsed_cnt = sum(1 for p in predicted_labels if p is not None)
                log(f"\n[Step {current_step_number} Summary]")
                log(f"GT: {gt_label} | Parsed: {parsed_cnt}/{N_SAMPLES_PER_STEP} | Match: {matches} | Conf: {confidence:.2f}")

                verification_count = 0
                for c_idx, chunk in enumerate(updated_cot_chunks):
                    if is_verification_chunk(chunk):
                        if verification_count == step_idx:
                            updated_cot_chunks[c_idx] = chunk + f"<confidence>{confidence:.2f}</confidence>"
                            break
                        verification_count += 1
            
            updated_item = item.copy()
            updated_item['cot_chunks'] = updated_cot_chunks
            updated_item['cot'] = ''.join(updated_cot_chunks)
            result_data.append(updated_item)
            
        except Exception as e:
            log(f"⚠️ Problem {current_idx} Error: {e}")
            log(traceback.format_exc())
            result_data.append(item)
            
        if (len(result_data) % args.save_interval == 0) or (i == len(target_data) - 1):
            log(f"💾 Checkpoint saved... ({len(result_data)} items)")
            try:
                with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
            except Exception as save_err:
                log(f"❌ Save Error: {save_err}", console=True)

    log("✅ 모든 작업 완료.", console=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())