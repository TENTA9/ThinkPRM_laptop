"""
RLHF Flow Process Reward Model

This module provides an RLHF-based PRM implementation using the RLHFlow model
for evaluating step-by-step reasoning processes.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class RLHFFlowPRM:
    """RLHF Flow Process Reward Model implementation."""
    
    def __init__(self,
                device: str = 'cuda',
                ) -> None:
        self.device = device
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(
        self, **model_kwargs
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
        )
        model = AutoModelForCausalLM.from_pretrained(
            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        ).eval()
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        plus_tag_id = tokenizer.encode("+")[-1]
        minus_tag_id = tokenizer.encode("-")[-1]
        self.candidate_tokens = [plus_tag_id, minus_tag_id]

        return model, tokenizer

    def predict_correctness_batch(
        self, questions: list[str], prefix_steps_batch: list[list[str]], batch_size: int = 2
    ) -> list[tuple[float, dict]]:
        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.
        
        special_tok_id = self.tokenizer("ки", return_tensors="pt").input_ids[0, 1]
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        conversations2 = []
        for question, prefix_steps in zip(questions, prefix_steps_batch, strict=True):
            conversation = []
            conversation2 = []
            for k, step in enumerate(prefix_steps):
                if k == 0:
                    text = question + " " + step
                else:
                    text = step
                conversation.append({"content": text, "role": "user"})
                conversation.append({"content": "+", "role": "assistant"})

                # we track to location of the special token with ки in order to extract the scores
                conversation2.append({"content": text, "role": "user"})
                conversation2.append({"content": "ки", "role": "assistant"})

            conversations.append(conversation)
            conversations2.append(conversation2)

        results = []
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            convs2_batch = conversations2[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                convs_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            inputs2_batch = self.tokenizer.apply_chat_template(
                convs2_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            assert inputs_batch.shape == inputs2_batch.shape
            with torch.no_grad():
                logits = self.model(inputs_batch).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[
                    :, :, 0
                ]  # 0 means the prob of + (1 mean -)

                for j in range(len(convs_batch)):
                    # We slice on the N-1 token since the model is trained to predict the Nth one ("+" in this case)
                    step_scores = scores[j, :-1][
                        inputs2_batch[j, 1:] == special_tok_id
                    ].tolist()
                    
                    prefix_score = step_scores[-1] # last step score is the full prefix score
                    step_labels = [1 if score > 0.5 else 0 for score in step_scores]
                    step_cots = [""] * len(step_labels)
                    
                    info = {
                        'full_prefix_score': prefix_score,
                        'step_scores': step_scores,
                        'step_cots': step_cots,
                        'step_labels': step_labels,
                        'input_text': self.tokenizer.decode(inputs_batch[j]),
                        'output_texts': [""],
                    }
                    
                    results.append((prefix_score, info))

        return results
