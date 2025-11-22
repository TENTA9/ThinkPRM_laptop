"""
Discriminative Process Reward Model

This module provides a discriminative PRM implementation that predicts correctness
by analyzing step-by-step reasoning processes.
"""

from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset.prm_dataset import PRMTrajectoryDataset
import re


class DiscriminativePRM:
    """Discriminative PRM implementation."""
    
    def __init__(self,
                model_name_or_path: str,
                step_sep: str = ' ки',
                pos_label_step_token: str = '+',
                neg_label_step_token: str = '-',
                random_reward: bool = False,
                max_length: int = 1024,
                device: str = 'cuda',
                batch_size: int = 8,
                long_cot: bool = False,
                **kwargs
                ) -> None:
        super().__init__()
        
        print("Loading PRM model from {}".format(model_name_or_path))
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16).to(device).eval() # bf16/fp16 might lead to inconsistent results
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.step_sep_id = self.tokenizer.encode(step_sep, add_special_tokens=False)[-1]
        self.pos_step_id = self.tokenizer.encode(pos_label_step_token, add_special_tokens=False)[-1]
        self.neg_step_id = self.tokenizer.encode(neg_label_step_token, add_special_tokens=False)[-1]
        self.random_reward = random_reward
        self.max_length = max_length
        self.batch_size = batch_size
        self.long_cot = long_cot

    def predict_correctness(self, question: str, prefix_steps: list[str]) -> tuple[float, dict]:
        # Tokenize the input
        inputs = self.process_example(question, prefix_steps)
        
        if inputs['input_ids'][-1] != self.step_sep_id:
            print("Warning: step separator not found in the input ids, adding it...")
            inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.tensor([self.step_sep_id])])
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.tensor([1])])

        input_ids = inputs['input_ids'].unsqueeze(0)  # Add batch dimension
        attention_mask = inputs['attention_mask'].unsqueeze(0)
        
        # Move tensors to the same device as the model
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        candidate_tokens = [self.pos_step_id, self.neg_step_id]

        # Get model outputs
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0] # 1 x seq_len

            step_scores = scores[input_ids == self.step_sep_id] # 1 x 
            step_scores = step_scores.cpu().tolist()

        full_prefix_score = step_scores[-1]
        step_labels = [1 if score > 0.5 else 0 for score in step_scores]
        step_cots = [""] * len(step_labels)
                
        info = {
            'full_prefix_score': full_prefix_score,
            'step_scores': step_scores,
            'step_cots': step_cots,
            'step_labels': step_labels,
            'input_text': inputs['input_text'],
            'output_texts': [""],
        }

        return full_prefix_score, info

    def predict_correctness_batch(self, questions: list[str], prefix_steps_list: list[list[str]]) -> list[tuple[float, dict]]:
        # Tokenize all inputs at once
        batch_inputs = []
        for question, prefix_steps in zip(questions, prefix_steps_list):
            inputs = self.process_example(question, prefix_steps)
            if inputs['input_ids'][-1] != self.step_sep_id:
                print("Warning: step separator not found in the input ids, adding it...")    
                inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.tensor([self.step_sep_id])])
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.tensor([1])])
            batch_inputs.append(inputs)

        # Pad sequences to max length in batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [inputs['input_ids'] for inputs in batch_inputs], 
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        ).to(self.model.device)
                
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [inputs['attention_mask'] for inputs in batch_inputs],
            batch_first=True,
            padding_value=0
        ).to(self.model.device)
        

        candidate_tokens = [self.pos_step_id, self.neg_step_id]
        
        # Get model outputs for entire batch at once
        with torch.inference_mode():
            logits = self.model(input_ids, attention_mask=attention_mask).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]  # batch_size x seq_len


        # Process results for each example
        results = []
        for i, prefix_steps in enumerate(prefix_steps_list):
            step_mask = (input_ids[i] == self.step_sep_id) & (attention_mask[i] == 1)  # Only consider valid positions
            step_scores = scores[i][step_mask].cpu().tolist()
            if len(step_scores) != len(prefix_steps):
                print("Warning: step scores and prefix steps are not of the same length. This is likely due to a very long chain.")
            full_prefix_score = step_scores[-1]
            
            step_labels = [1 if score > 0.5 else 0 for score in step_scores]
            step_cots = [""] * len(step_labels)
            
            info = {
                'full_prefix_score': full_prefix_score,
                'step_scores': step_scores,
                'step_cots': step_cots,
                'step_labels': step_labels,
                'input_text': self.tokenizer.decode(input_ids[i]),
                'output_texts': [""],
            }
                

            results.append((full_prefix_score, info))

        return results

    def process_example(self, question: str, prefix_steps: list[str]):
        # Prepare the example for tokenization
        
        example = {
            'question': question,
            'steps_with_labels': [{'step': step, 'label': '+'} for step in prefix_steps], # placeholder labels
            'solution_label': -1 # placeholder label
        }
                
        # Call tokenize_example from prm_dataset.py
        tokenized_example = PRMTrajectoryDataset.tokenize_example(
            example, 
            self.tokenizer, 
            self.step_sep_id, 
            self.pos_step_id, 
            self.neg_step_id, 
            self.max_length,
            config={},
            split='test',
            add_step_tokens=not self.long_cot
        )

        # Extract the required fields
        input_ids = tokenized_example['input_ids']
        attention_mask = tokenized_example['attention_mask']
                
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
    def score(self, questions: list[str], outputs: list[list[str]], batch_size: int = 8):
        """ to be used with beam search/best of n"""
        
        # Flatten inputs
        flat_questions = []
        flat_completions = []
        for question, answers in zip(questions, outputs):
            for answer in answers:
                flat_questions.append(question)
                flat_completions.append(answer)
                
        # Process completions
        if not self.long_cot:
            flat_completions = [completion.replace("## Step", "Step").strip() for completion in flat_completions]
            flat_completions = [re.split(r'Step \d+:', completion) for completion in flat_completions]
            flat_completions = [[s.strip() for s in completion if s.strip()] for completion in flat_completions]
        else:
            #### long cots will not have step separators. So treat them as a single step
            flat_completions = [[completion.strip()] for completion in flat_completions]
                            
        # Run inference in batches
        flat_results = []
        for i in range(0, len(flat_questions), batch_size):
            batch_questions = flat_questions[i:i+batch_size]
            batch_completions = flat_completions[i:i+batch_size]
            batch_results = self.predict_correctness_batch(batch_questions, batch_completions)
            flat_results.extend(batch_results)
            
        assert len(flat_results) == len(flat_questions), f"Number of results {len(flat_results)} does not match number of questions {len(flat_questions)}"
            
        # Reshape results to match input shape
        scores = []
        idx = 0
        for answers in outputs:
            answer_scores = []
            for _ in answers:
                res_info_tuple = flat_results[idx]
                answer_scores.append([res_info_tuple[0]])
                idx += 1
            scores.append(answer_scores)
            
        return scores
