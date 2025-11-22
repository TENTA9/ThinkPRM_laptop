"""
단순 테스트: 두 번째 문제의 Step 3만 검증
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
DEBUG_LOG_FILENAME = "debug_step3.log"

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
    """
    step_index번째 검증 청크 직전까지의 모든 청크를 연결
    
    Args:
        cot_chunks: 전체 cot_chunks 리스트
        step_index: 검증하려는 스텝의 인덱스 (0-based, Step 3이면 2)
    
    Returns:
        해당 스텝 검증 청크 직전까지의 모든 청크를 연결한 문자열
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
    
    # cot_prefix를 이미 생성된 것처럼 추가
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
def generate_step_verification(s, prompt: str):
    """특정 스텝에 대한 검증 CoT 생성 (1회)"""
    stop_patterns = [
        "\n\nStep ",  # 다음 스텝이 시작되면 중단
        "<|im_end|>",
        "</s>",
        "</think>"
    ]
    
    s += prompt
    s += gen(
        "step_verification",
        max_tokens=MAX_GENERATION_TOKENS,
        temperature=0.6,
        stop=stop_patterns
    )
    
    return s

# ============================================================================
# 메인 함수
# ============================================================================

def main():
    global log_file
    
    # 로그 파일 열기
    log_file = open(DEBUG_LOG_FILENAME, 'w', encoding='utf-8')
    
    log("=" * 70)
    log("Step 3 검증 테스트")
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
        
        # 두 번째 문제 (인덱스 1)
        item = data[1]
        log("      ✓ 로드 성공")
        log(f"\n문제: {item['problem'][:100]}...")
        log(f"Step 개수: {item['valid_prefix_step_count']}")
    except Exception as e:
        log(f"      ❌ 로드 실패: {e}")
        log_file.close()
        return 1

    # 4. Step 3 검증
    log(f"\n[4/4] Step 3 검증 중...")
    
    try:
        problem = item['problem']
        prefix = item['prefix']
        cot_chunks = item['cot_chunks']
        gt_step_labels = item['gt_step_labels']
        
        # Step 3은 인덱스 2 (0-based)
        step_idx = 2
        
        # Step 3 직전까지의 CoT 프리픽스 생성
        cot_prefix = get_prefix_before_step(cot_chunks, step_idx)
        
        log(f"\n{'=' * 70}")
        log("=== CoT Prefix (Step 3 직전까지) ===")
        log(f"{'=' * 70}")
        log(cot_prefix)
        log(f"{'=' * 70}")
        log("=== CoT Prefix 끝 ===")
        log(f"{'=' * 70}\n")
        
        # 프롬프트 생성
        prompt_text = format_verification_prompt(
            tokenizer,
            problem=problem,
            prefix=prefix,
            cot_prefix=cot_prefix
        )
        
        log(f"프롬프트 길이: {len(prompt_text)} 문자\n")
        
        log(f"{'=' * 70}")
        log("=== 전체 프롬프트 ===")
        log(f"{'=' * 70}")
        log(prompt_text)
        log(f"{'=' * 70}")
        log("=== 전체 프롬프트 끝 ===")
        log(f"{'=' * 70}\n")
        
        # 생성
        log("생성 시작...")
        state = generate_step_verification.run(prompt=prompt_text)
        generated_text = state["step_verification"]
        
        log(f"\n{'=' * 70}")
        log("=== 생성된 Step 3 검증 ===")
        log(f"{'=' * 70}")
        log(generated_text)
        log(f"{'=' * 70}")
        log("=== 생성 끝 ===")
        log(f"{'=' * 70}\n")
        
        # 라벨 추출
        pred_label = extract_boxed_label(generated_text)
        gt_label = gt_step_labels[step_idx]
        gt_numeric = 1 if gt_label == '+' else 0
        
        log(f"GT 라벨: {gt_label} ({gt_numeric})")
        log(f"예측 라벨: {pred_label}")
        log(f"일치 여부: {'✓' if pred_label == gt_numeric else '✗'}")
        
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