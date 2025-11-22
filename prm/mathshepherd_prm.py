"""
MathShepherd Process Reward Model

This module provides a specialized PRM implementation for math problems
using the MathShepherd model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class MathShepherdPRM:
    """MathShepherd Process Reward Model implementation."""
    
    def __init__(self,
                device: str = 'cuda',
                ) -> None:
        
        good_token = '+'
        bad_token = '-'
        step_tag = 'ки'

        self.tokenizer = AutoTokenizer.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm')
        self.candidate_tokens = self.tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]


        print("Loading PRM model from peiyi9979/math-shepherd-mistral-7b-prm")
        self.model = AutoModelForCausalLM.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm', torch_dtype=torch.bfloat16).eval()
        self.device = device
        self.model.to(self.device)
        
        self.step_tag_id = self.tokenizer.encode(f"{step_tag}")[-1] # 12902
        self.step_tag = step_tag

    def predict_correctness(self, question: str, prefix_steps: list[str]) -> tuple[float, dict]:
        output = ""
        for i, step in enumerate(prefix_steps, 1):
            output += f"Step {i}: {step} {self.step_tag}\n"
        
        output = output.strip()
        input_for_prm = f"{question} {output}"
        input_ids = torch.tensor([self.tokenizer.encode(input_for_prm)]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids).logits[:, :, self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0] # 1 x seq_len
            step_scores = scores[input_ids == self.step_tag_id] # 1 x 

        step_scores = step_scores.cpu().tolist()

        if len(step_scores) != len(prefix_steps):
            print("warning: something probably wrong happened with tokenization that add/removed a step tag")

        prefix_score = step_scores[-1]
        step_labels = [1 if score > 0.5 else 0 for score in step_scores]
        step_cots = [""] * len(step_labels)
        
        info = {
            'full_prefix_score': prefix_score,
            'step_scores': step_scores,
            'step_cots': step_cots,
            'step_labels': step_labels,
            'input_text': input_for_prm,
            'output_texts': [""],
        }

        return prefix_score, info


    def predict_correctness_batch(self, questions: list[str], prefix_steps_batch: list[list[str]]) -> list[tuple[float, dict]]:
        # Process each example into formatted input string
        batch_inputs = []
        for question, prefix_steps in zip(questions, prefix_steps_batch):
            output = ""
            for i, step in enumerate(prefix_steps, 1):
                output += f"Step {i}: {step} {self.step_tag}\n"
            output = output.strip()
            
            ###### MATHShepherd expects the answer format: 'The answer is: <answer>'
            output = output.replace('The answer is', 'The answer is:').replace('the answer is', 'The answer is:').replace('Final Answer: The final answer is', 'The answer is:')
            batch_inputs.append(f"{question} {output}")

        # Tokenize all inputs
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        input_ids = self.tokenizer(batch_inputs, padding=True, return_tensors="pt").input_ids.to(self.device)
                
        # Get model predictions
        with torch.no_grad():
            logits = self.model(input_ids).logits[:, :, self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]  # batch_size x seq_len
        
        # Extract scores for each step tag
        results = []
        for i, prefix_steps in enumerate(prefix_steps_batch):
            step_mask = (input_ids[i] == self.step_tag_id)
            step_scores = scores[i][step_mask].cpu().tolist()
            
            if len(step_scores) != len(prefix_steps):
                print("warning: something probably wrong happened with tokenization that add/removed a step tag")
            
            prefix_score = step_scores[-1] # last step score is the full prefix score
            step_labels = [1 if score > 0.5 else 0 for score in step_scores]
            step_cots = [""] * len(step_labels)
            
            info = {
                'full_prefix_score': prefix_score,
                'step_scores': step_scores,
                'step_cots': step_cots,
                'step_labels': step_labels,
                'input_text': batch_inputs[i],
                'output_texts': [""],
            }
            
            results.append((prefix_score, info))
            
        return results
