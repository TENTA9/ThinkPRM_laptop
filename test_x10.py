"""
테스트: 두 번째 문제의 Step 3을 10번 검증
"""

import json
import re
from transformers import AutoTokenizer
from sglang import function, gen, set_default_backend, RuntimeEndpoint

# ============================================================================
# 설정
# ============================================================================
SGLANG_ENDPOINT = "http://127.0.0.1:31111"
MODEL_NAME_OR_PATH = "KirillR/QwQ-32B-Preview-AWQ"
MAX_GENERATION_TOKENS = 2048
INPUT_FILENAME = "thinkprm_data.json"
DEBUG_LOG_FILENAME = "debug_step3_x10.log"
N_SAMPLES = 10

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
    """검증 청크인지 확인"""
    chunk = chunk.strip()
    if not chunk.startswith("Step"):
        return False
    if "\\boxed{" not in chunk:
        return False
    return True

def get_prefix_before_step(cot_chunks, step_index):
    """step_index번째 검증 청크 직전까지의 모든 청크를 연결"""
    prefix_chunks = []
    verification_count = 0
    
    for chunk in cot_chunks:
        if is_verification_chunk(chunk):
            if verification_count == step_index:
                break
            verification_count += 1
        prefix_chunks.append(chunk)
    
    return ''.join(prefix_chunks)

def format_verification_prompt(tokenizer, problem, prefix, cot_prefix):
    """검증 프롬프트 생성"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": f"""You are given a math problem and a proposed step-by-step solution:

[Math Problem]

{problem}

[Solution]

{prefix}

Review and critique each step in the proposed solution.

**CRITICAL FORMAT REQUIREMENT:**
For EVERY step, you MUST conclude your analysis with EXACTLY ONE of these:
- \\boxed{{correct}} if the step is correct
- \\boxed{{incorrect}} if the step is wrong

**Example Format:**
Step 1: [Your analysis of step 1]
\\boxed{{correct}}

Step 2: [Your analysis of step 2]
\\boxed{{incorrect}}

If the solution is incomplete, only verify the provided steps.

IMPORTANT: Do NOT provide a summary conclusion. Judge EACH step individually."""
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_prompt = prompt + cot_prefix
    
    return full_prompt

def extract_boxed_label(text):
    """생성된 텍스트에서 \\boxed{correct} 또는 \\boxed{incorrect} 추출"""
    pattern = r'\\boxed\{(correct|incorrect)\}'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        label = match.group(1).lower()
        return 1 if label == "correct" else 0
    return None

# ============================================================================
# SGLang 생성 함수
# ============================================================================

prompt_to_states = {}

@function
def generate_step_verification(s, prompt: str, n: int):
    """특정 스텝에 대한 검증 CoT 생성 (n회)"""
    stop_patterns = [
        "\n\nStep ",
        "<|im_end|>",
        "</s>",
        "</think>"
    ]
    
    s += prompt
    forks = s.fork(n)
    
    for fork in forks:
        fork += gen(
            "step_verification",
            max_tokens=MAX_GENERATION_TOKENS,
            temperature=0.6,
            stop=stop_patterns
        )
        
        if prompt not in prompt_to_states:
            prompt_to_states[prompt] = []
        prompt_to_states[prompt].append(fork)

# ============================================================================
# 메인 함수
# ============================================================================

def main():
    global log_file
    
    # 로그 파일 열기
    log_file = open(DEBUG_LOG_FILENAME, 'w', encoding='utf-8')
    
    log("=" * 70)
    log(f"Step 3 검증 테스트 (x{N_SAMPLES})")
    log("=" * 70)
    
    # 1. SGLang 서버 연결
    log(f"\n[1/4] SGLang 서버 연결 중...")
    try:
        set_default_backend(RuntimeEndpoint(SGLANG_ENDPOINT))
        log("      ✓ 연결 성공")
    except Exception as e:
        log(f"      ❌ 연결 실패: {e}")
        log_file.close()
        return 1

    # 2. 토크나이저 로드
    log(f"\n[2/4] 토크나이저 로드 중...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        log("      ✓ 로드 성공")
    except Exception as e:
        log(f"      ❌ 로드 실패: {e}")
        log_file.close()
        return 1

    # 3. 데이터 로드 (두 번째 문제만)
    log(f"\n[3/4] 데이터 로드 중...")
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        item = data[1]
        log("      ✓ 로드 성공")
        log(f"\n문제: {item['problem'][:100]}...")
        log(f"Step 개수: {item['valid_prefix_step_count']}")
    except Exception as e:
        log(f"      ❌ 로드 실패: {e}")
        log_file.close()
        return 1

    # 4. Step 3 검증 (10번)
    log(f"\n[4/4] Step 3 검증 중 (x{N_SAMPLES})...")
    
    try:
        problem = item['problem']
        prefix = item['prefix']
        cot_chunks = item['cot_chunks']
        gt_step_labels = item['gt_step_labels']
        
        step_idx = 2  # Step 3
        
        # CoT 프리픽스 생성
        cot_prefix = get_prefix_before_step(cot_chunks, step_idx)
        
        # 프롬프트 생성
        prompt_text = format_verification_prompt(
            tokenizer,
            problem=problem,
            prefix=prefix,
            cot_prefix=cot_prefix
        )
        
        log(f"프롬프트 길이: {len(prompt_text)} 문자\n")
        
        # prompt_to_states 초기화
        global prompt_to_states
        prompt_to_states = {}
        
        # 10번 생성
        log(f"생성 시작 (x{N_SAMPLES})...")
        batch_args = [{'prompt': prompt_text, 'n': N_SAMPLES}]
        _ = generate_step_verification.run_batch(batch_args)
        
        # 생성된 결과 수집
        generated_verifications = []
        for state in prompt_to_states[prompt_text]:
            verification = state["step_verification"]
            generated_verifications.append(verification.strip())
        
        log(f"\n생성 완료! {len(generated_verifications)}개 샘플\n")
        
        # 각 생성 결과 분석
        log("=" * 70)
        log("=== 생성된 검증들 ===")
        log("=" * 70)
        
        predicted_labels = []
        for i, gen_text in enumerate(generated_verifications):
            log(f"\n[샘플 {i+1}]")
            log("-" * 70)
            log(gen_text)
            log("-" * 70)
            
            pred_label = extract_boxed_label(gen_text)
            predicted_labels.append(pred_label)
            log(f"추출된 라벨: {pred_label} ({'correct' if pred_label == 1 else 'incorrect' if pred_label == 0 else 'PARSE_FAILED'})")
        
        # GT와 비교
        gt_label = gt_step_labels[step_idx]
        gt_numeric = 1 if gt_label == '+' else 0
        
        log("\n" + "=" * 70)
        log("=== 결과 분석 ===")
        log("=" * 70)
        log(f"GT 라벨: {gt_label} ({gt_numeric})")
        log(f"\n예측 라벨들:")
        
        valid_predictions = [p for p in predicted_labels if p is not None]
        if valid_predictions:
            matches = sum(1 for pred in valid_predictions if pred == gt_numeric)
            confidence = matches / len(valid_predictions)
            
            log(f"  - 총 {len(predicted_labels)}개 생성")
            log(f"  - 파싱 성공: {len(valid_predictions)}개")
            log(f"  - GT와 일치: {matches}개")
            log(f"  - Confidence: {confidence:.2f}")
        else:
            log("  - 파싱 성공한 예측 없음!")
        
        log("\n" + "=" * 70)
        log("테스트 완료!")
        log("=" * 70)
        log(f"\n디버그 로그 저장됨: {DEBUG_LOG_FILENAME}")
        
        log_file.close()
        return 0
        
    except Exception as e:
        log(f"      ❌ 오류 발생: {e}")
        import traceback
        log(traceback.format_exc())
        log_file.close()
        return 1

if __name__ == "__main__":
    import os
    exit_code = main()
    
    print("정리 중...")
    os._exit(exit_code)