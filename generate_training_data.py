"""
ThinkPRM Training Data Generation Script

이 스크립트는 논문 "Process Reward Models That Think"의 데이터 생성 과정을 재현합니다.

사용법:
1. 터미널 1에서 SGLang 서버 실행:
   CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 python -m sglang.launch_server \
       --model-path "casperhansen/Llama-3.2-3B-Instruct-AWQ" \
       --quantization awq --port 31111 --host 127.0.0.1

2. 터미널 2에서 이 스크립트 실행:
   python generate_training_data.py
"""

import json
import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer
from sglang import function, gen, set_default_backend, RuntimeEndpoint
from tqdm import tqdm
import traceback

# 경로 설정
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# utils.helper에서 프롬프트 포맷 함수 import
try:
    from utils.helper import format_verification_cot_for_thinkprm
except ImportError as e:
    print(f"❌ 오류: utils.helper 모듈을 찾을 수 없습니다: {e}")
    print("현재 디렉토리가 thinkprm 루트인지 확인하세요.")
    sys.exit(1)

def format_verification_cot_with_step_labels(tokenizer, problem, solution):
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

{solution}

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
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ============================================================================
# 설정
# ============================================================================
SGLANG_ENDPOINT = "http://127.0.0.1:31111"
MODEL_NAME_OR_PATH = "KirillR/QwQ-32B-Preview-AWQ"
N_SAMPLES_PER_PROBLEM = 4  # 문제당 생성할 verification CoT 개수
MAX_GENERATION_TOKENS = 2048  # AWQ 모델용 토큰 제한
OUTPUT_FILENAME = "all_outputs.json"
N_PROBLEMS = 10  # None으로 설정하면 전체 데이터셋 처리

# ============================================================================
# SGLang 생성 함수
# ============================================================================

# 전역 딕셔너리: 프롬프트별로 생성된 state 저장
prompt_to_states = {}

@function
def cot_eval(s, prompt: str, n: int):
    """
    Verification CoT 생성 함수
    - thinkprm_api.py의 방식을 정확히 따름
    """
    stop_patterns = [
        "Is the solution correct?",
        "<|im_end|>",
        "</s>",
        "</think>"
    ]
    
    s += prompt
    forks = s.fork(n)
    
    for fork in forks:
        fork += gen(
            "verification",  # 변수명을 "verification"으로 통일
            max_tokens=MAX_GENERATION_TOKENS,
            temperature=0.6,
            stop=stop_patterns
        )
        # 각 fork를 저장
        if prompt not in prompt_to_states:
            prompt_to_states[prompt] = []
        prompt_to_states[prompt].append(fork)

# ============================================================================
# 메인 함수
# ============================================================================
def main():
    print("=" * 70)
    print("ThinkPRM Training Data Generation")
    print("=" * 70)
    
    # 1. SGLang 서버 연결
    print(f"\n[1/5] SGLang 서버 연결 중...")
    print(f"      Endpoint: {SGLANG_ENDPOINT}")
    
    try:
        set_default_backend(RuntimeEndpoint(SGLANG_ENDPOINT))
        print("      ✓ 연결 성공")
    except Exception as e:
        print(f"      ❌ 연결 실패: {e}")
        print("\n서버를 먼저 실행하세요:")
        print("CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 python -m sglang.launch_server \\")
        print(f"    --model-path \"{MODEL_NAME_OR_PATH}\" \\")
        print("    --quantization awq --port 31111 --host 127.0.0.1")
        return 1

    # 2. 토크나이저 로드
    print(f"\n[2/5] 토크나이저 로드 중...")
    print(f"      Model: {MODEL_NAME_OR_PATH}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        print("      ✓ 로드 성공")
    except Exception as e:
        print(f"      ❌ 로드 실패: {e}")
        return 1

    # 3. PRM800K 데이터셋 로드
    print(f"\n[3/5] PRM800K 데이터셋 로드 중...")
    
    try:
        data_files = {
            "train": "https://github.com/openai/prm800k/raw/refs/heads/main/prm800k/data/phase1_train.jsonl",
            "test": "https://github.com/openai/prm800k/raw/refs/heads/main/prm800k/data/phase1_test.jsonl",
        }
        
        dataset = load_dataset("json", data_files=data_files, split="train")
        print(f"      ✓ 로드 성공: {len(dataset):,}개 샘플")
        
        # 테스트용 서브셋 선택
        if N_PROBLEMS is not None:
            dataset = dataset.select(range(min(N_PROBLEMS, len(dataset))))
            print(f"      ℹ 테스트용 {len(dataset)}개 샘플 선택")
            
    except Exception as e:
        print(f"      ❌ 로드 실패: {e}")
        traceback.print_exc()
        return 1

    # 4. Verification CoT 생성
    print(f"\n[4/5] Verification CoT 생성 중...")
    print(f"      - 문제 수: {len(dataset)}")
    print(f"      - 문제당 생성 개수: {N_SAMPLES_PER_PROBLEM}")
    print(f"      - 총 생성 목표: {len(dataset) * N_SAMPLES_PER_PROBLEM}개\n")
    
    all_outputs = []
    n_skipped = 0
    
    for example in tqdm(dataset, desc="생성 중"):
        try:
            # 문제 추출
            problem = example['question']['problem']
            
            # 솔루션 재구성
            solution_parts = []
            gt_labels = []
            
            for step_data in example['label']['steps']:
                chosen_idx = step_data.get('chosen_completion')
                
                if chosen_idx is not None and chosen_idx >= 0:
                    if 'completions' in step_data and len(step_data['completions']) > chosen_idx:
                        completion = step_data['completions'][chosen_idx]
                        solution_parts.append(completion['text'])
                        rating = completion.get('rating', 0)
                        gt_labels.append(1 if rating == 1 else 0)
                        
                elif step_data.get('human_completion') is not None:
                    human_comp = step_data['human_completion']
                    if isinstance(human_comp, dict) and 'text' in human_comp:
                        solution_parts.append(human_comp['text'])
                        gt_labels.append(1)
                    elif isinstance(human_comp, str):
                        solution_parts.append(human_comp)
                        gt_labels.append(1)
            
            prefix = '\n'.join(solution_parts)
            
            # 유효성 검사
            if not all([problem, prefix, gt_labels]):
                n_skipped += 1
                continue

            # 프롬프트 생성
            prompt_text = format_verification_cot_with_step_labels(
                tokenizer,
                problem=problem,
                solution=prefix
            )
            
            # prompt_to_states 초기화
            global prompt_to_states
            prompt_to_states = {}
            
            # Verification CoT 생성
            batch_args = [{'prompt': prompt_text, 'n': N_SAMPLES_PER_PROBLEM}]
            _ = cot_eval.run_batch(batch_args)
            
            # 생성된 CoT 수집 (thinkprm_api.py 방식)
            generated_cots_list = []
            for state in prompt_to_states[prompt_text]:
                cot = state["verification"]
                generated_cots_list.append(cot.strip())
            
            # 저장
            output_data = {
                "problem": problem,
                "prefix": prefix,
                "traj_gt_labels": gt_labels,
                "prompt": prompt_text,
                "generations": generated_cots_list
            }
            all_outputs.append(output_data)
            
        except Exception as e:
            print(f"\n⚠️  오류 발생 (건너뜀): {e}")
            import traceback
            traceback.print_exc()
            n_skipped += 1
            continue
    
    # 5. 결과 저장
    print(f"\n[5/5] 결과 저장 중...")
    
    if not all_outputs:
        print("      ❌ 생성된 데이터가 없습니다.")
        return 1
    
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(all_outputs, f, indent=2, ensure_ascii=False)
        
        # 통계 계산
        total_generations = sum(len(item['generations']) for item in all_outputs)
        avg_cot_length = sum(
            len(cot) for item in all_outputs for cot in item['generations']
        ) / total_generations
        avg_gens_per_problem = total_generations / len(all_outputs)
        
        print(f"      ✓ 저장 완료: {OUTPUT_FILENAME}")
        print("\n" + "=" * 70)
        print("생성 완료!")
        print("=" * 70)
        print(f"처리된 문제 수:        {len(all_outputs):,}개")
        print(f"건너뛴 문제 수:        {n_skipped:,}개")
        print(f"총 생성된 CoT:         {total_generations:,}개")
        print(f"문제당 평균 생성:      {avg_gens_per_problem:.1f}개")
        print(f"평균 CoT 길이:         {avg_cot_length:,.0f} 문자")
        print("=" * 70)
        
        # 명시적 종료
        print("\n프로그램을 종료합니다...")
        return 0
        
    except Exception as e:
        print(f"      ❌ 저장 실패: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import os
    exit_code = main()
    
    # 강제 종료 (SGLang 백엔드 연결 정리)
    print("정리 중...")
    os._exit(exit_code)  # 모든 스레드 즉시 종료