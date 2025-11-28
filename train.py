import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
    BitsAndBytesConfig # 👈 4bit 지원을 위한 모듈 추가
)
from peft import LoraConfig, get_peft_model, TaskType

# 사용자의 dataset 폴더 구조에 맞춰 import
from dataset.prm_dataset import LongThoughtCritiqueDataset

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-1.5B-Instruct", 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})
    lora_r: int = field(default=32, metadata={"help": "LoRA r dimension."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA"}
    )
    # 👇 이 줄이 있어야 --load_in_4bit 옵션을 인식합니다!
    load_in_4bit: bool = field(default=False, metadata={"help": "Load model in 4-bit precision (QLoRA)."})

@dataclass
class DataArguments:
    data_dir: str = field(
        default=None, 
        metadata={"help": "Path to the folder containing training data json files."}
    )
    max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length."}
    )
    # Dataset Class 설정
    max_cots_per_solution: int = field(default=1, metadata={"help": "Max CoTs per solution."})
    match_all_step_labels: bool = field(default=True, metadata={"help": "Filter based on all step labels."})
    filter_based_on_length: bool = field(default=True, metadata={"help": "Filter out too long sequences."})
    balance_data: bool = field(default=False, metadata={"help": "Balance correct/incorrect examples."})
    add_think_token: bool = field(default=True, metadata={"help": "Add <think> token explicitly."})
    
    # 호환성 유지용 더미 속성들
    train_with_gold_solutions: bool = False
    add_partial_prefixes: bool = False
    single_label: bool = False 
    direct_prm: bool = False
    cot_incorrect_only: bool = False

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 로깅 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    set_seed(training_args.seed)

    # 1. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if data_args.add_think_token and "<think>" not in tokenizer.vocab:
        special_tokens_dict = {'additional_special_tokens': ['<think>', '</think>']}
        tokenizer.add_special_tokens(special_tokens_dict)

    # 2. 데이터셋 준비
    logger.info("Loading dataset...")
    train_dataset = LongThoughtCritiqueDataset(
        data_path=data_args.data_dir,
        tokenizer=tokenizer,
        config=data_args, 
        split='train'
    )

    # 3. 모델 로드
    logger.info("Loading model...")
    
    # QLoRA 설정 (4-bit 사용 시에만 적용됨)
    quantization_config = None
    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        logger.info("🚀 Using QLoRA (4-bit quantization)")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config, # 설정 적용 (None이면 무시됨)
        use_cache=False if training_args.gradient_checkpointing else True
    )

    model.resize_token_embeddings(len(tokenizer))
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # 4. LoRA 설정 (사용 안 하면 패스)
    if model_args.use_lora:
        logger.info("Applying LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # 5. Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=train_dataset.collate_fn,
    )

    # 6. 학습 시작
    logger.info("Starting training...")
    trainer.train()

    # 7. 모델 저장
    logger.info("Saving model...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()