import json
import re
import time
import torch

from api_generate_copy.similar_classification import SimilarClassification

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_MIN_P = 0

class Qwen3Pipeline:
    """
    Choice pipeline for QWen3
    """
    classifier: SimilarClassification
    embedding_model: any

    def __init__(self, conceptual_index_path, computational_index_path, embedding_model):
        self.classifier = SimilarClassification(conceptual_index_path=conceptual_index_path, computational_index_path=computational_index_path)
        self.embedding_model = embedding_model
        pass

    def generate_prompt(self, template, dict):
        prompt = template

        for key, val in dict.items():
            prompt = prompt.replace(f'<?{key}>', val)
        
        return prompt
    
    def create_conceptual_prompt(self, premises, question):
        dict = {}
        
        premise_list = [{ 'idx': (i+1), 'premise': premises[i] } for i in range(len(premises))]

        # <?premises>
        premise_vals = ""

        for premise_item in premise_list:
            premise_idx = premise_item['idx']
            premise = premise_item['premise']

            premise_vals += f"    - Premise {premise_idx}: {premise} \n"
        dict['premises'] = premise_vals

        dict['question'] = question
        return self.generate_prompt(PROMPT_INPUT_INFO, dict=dict)

    def generate_answer(self, tokenizer, model, messages, max_new_tokens = 32000, enable_thinking: bool=False):
        with torch.no_grad():
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking = enable_thinking
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            input_ids = model_inputs.input_ids

            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                temperature=DEFAULT_TEMPERATURE,
                top_k=DEFAULT_TOP_K,
                top_p=DEFAULT_TOP_P,
                min_p=DEFAULT_MIN_P
            )

        # Decode output (including both prompt and generated text)
        generated_sequence = output.sequences[0]
        full_generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        
        # Get only generated text (remove prompt)
        input_text_len = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
        return full_generated_text[input_text_len:].strip()

    def parse_response(self, llm_response):
        """
        Detect response including `answer`, `idx`, `explanation` in XML element <answer>, <idx>, <explanation>
        """
        
        # original llm resonse

        new_answer = {'answer': '', 'idx': [], 'explanation': '' }

        answer_start = llm_response.find('<answer>')
        answer_end = llm_response.find('</answer>')
        idx_start = llm_response.lower().find('<idx>')
        idx_end = llm_response.lower().find('</idx>')
        explanation_start = llm_response.find('<explanation>')
        explanation_end = llm_response.find('</explanation>')

        if answer_start > 0 and answer_end > 0 and answer_start < answer_end :
            answer_raw_str = llm_response[answer_start: answer_end].replace('<answer>', '').replace('</answer>', '').strip()
            new_answer['answer'] = answer_raw_str

        if idx_start> 0 and idx_end > 0 and idx_start < idx_end:
            idx_raw_str = llm_response[idx_start: idx_end].replace('<idx>', '').replace('</idx>', '').strip()
            idx = re.findall(r'\d+', idx_raw_str)
            idx = [int(id) for id in idx]
            new_answer['idx'] = list(set(idx))

        if explanation_start > 0 and explanation_end > 0 and explanation_start < explanation_end:
            explanation_raw_str = llm_response[explanation_start: explanation_end].replace('<explanation>', '').replace('</explanation>', '').strip()
            new_answer['explanation'] = explanation_raw_str

        return new_answer

    def run(self, premises, question, tokenizer, model, trace=False, enable_thinking: bool=False):
        # Step 1: classify question
        question_type, scores = self.classifier.classify(question=question, embedding_model=self.embedding_model)

        final_answer = { 'question': question, 'answer': '', 'idx': [], 'explanation': '', 'res': '' , 'error': '', 'manual': False }

        if question_type == 1:
            guildline = PROMPT_CONCEPTUAL_QUESTION_GUILDLINE
        else:
            guildline = PROMPT_COMPUTATIONAL_QUESTION_GUILDLINE

        start_time = time.perf_counter()
    
        # Propmt provider premises and question
        prompt = self.create_conceptual_prompt(premises=premises, question=question)
        messages = [
            {"role": "system", "content": guildline},
            {"role": "user", "content": prompt}
        ]
        if trace:
            print(guildline)
            print(prompt)
        response = self.generate_answer(tokenizer=tokenizer, model=model, messages=messages, enable_thinking=enable_thinking)
        end_time = time.perf_counter()
        final_answer['res'] = response
        cost = end_time - start_time
        if trace:
            print(response)
        try:
            parsed_res = self.parse_response(response)
            final_answer['answer'] = parsed_res['answer']
            final_answer['idx'] = parsed_res['idx']
            final_answer['explanation'] = parsed_res['explanation']
        except Exception as e:
            if trace:
                print('   Cannot parse response: ', str(e))
            final_answer['error'] = str(e)

        final_answer['details_time'] = cost
        
        return final_answer


PROMPT_CONCEPTUAL_QUESTION_GUILDLINE = """
You are a reasoning assistant trained to answer the question based on the given domain knowledge.
Given domain knowledge expressed in set of natural language premises and a question. Your task is **reason step by step** to answer the question  according to the strict requirements below.

**Strict Requirements**
  - Use ONLY pieces of information in the given knowledge. DO NOT introduce any external knowledge or assumptions.
  - Focus on step-by-step reasoning to answer the question. Clearly cite the source of every fact or condition used, refer to the premise number or quote the relevant part.
  - Do not include irrelevant premises or facts that are not directly required for answering the question
  - If a conclusion cannot be reached with certainty, explicitly state that it cannot be determined based on the available premises.

Steps to follow:
Step 1: Analyze the Premises:
  - For each premise, extract:
    - Key entities, facts, or definitions.
    - Logical conditions or rules (e.g., “if…then…”)
    - Do not perform inference in this step
Step 2: Perform step-by-step reasoning to answer the question.
  - Use the extracted facts to reason toward the answer.
  - At each step:
    - Clearly state what is being concluded.
    - Cite the exact premise(s) used or refer to earlier steps.
    - Avoid skipping logical steps or assuming information not provided.
Step 3: Finalize answer and put your response to following format:
```
<response>
    <answer>{answer}</answer>
    <idx>{idx}</idx>
    <explanation>{explanation}</explanation>
</response>
```
Field description:
  - `{answer}`: The final concise answer:
    - For multiple choice question, answer only is your choice letter. (e.g. 'A', 'B', 'C', 'D')
    - For Yes/No/Uncertain question, answer restricted in 'Yes' , 'No' , 'Uncertain'
    - For How many question, answer is a number.
    - For multiple-component question, each answer separated with comma (e.g. '120, No')
  - `{idx}`: A comma-separated list of the premise numbers (from `Premise #X`) that support the final answer.
  - `{explanation}`: Reasoning text **wrapped inside** <think> tags, written in natural language, clearly referring to the source premises (e.g., “From Premise 2, we know that…”).
"""

PROMPT_COMPUTATIONAL_QUESTION_GUILDLINE = """
You are a reasoning assistant trained to answer the question based on the given domain knowledge.
Given domain knowledge expressed in set of natural language premises and a question. Your task is **reason step by step** to answer the question  according to the strict requirements below.

**Strict Requirements**
  - Use ONLY pieces of information in the given knowledge. DO NOT introduce any external knowledge or assumptions.
  - Focus on step-by-step reasoning to answer the question. Clearly cite the source of information used, refer to the premise number or quote the relevant part.
  - Do not include irrelevant premises or facts that are not directly required for answering the question
  - If a conclusion cannot be reached with certainty, explicitly state that it cannot be determined based on the available premises.

Steps to follow:
Step 1: Analyze the question:
  - Identify its components (e.g., Yes/No, Multiple Choice, How Many) and their relationships.
  - For each component:
    - Identify the core concept(s) involved (e.g., GPA, Credits, etc)
    - Based on those concepts, extract relevant facts, definitions, logical conditions, or rules from the premises.
Step 2: Perform step-by-step reasoning to answer the question.
For each question components:
  - Use ONLY the provided information to reason toward the answer.
  - At each step:
    - Clearly state what is being concluded.
    - Cite the exact premise(s) used or refer to earlier steps.
    - Avoid skipping logical steps or assuming information not provided.
Step 3: Finalize answer and put your response to following format:
```
<response>
    <answer>{answer}</answer>
    <idx>{idx}</idx>
    <explanation>{explanation}</explanation>
</response>
```
Field description:
  - `{answer}` is final concise answer. For instance:
    + For multiple choice question, your answer must be a single letter. (e.g. 'A', 'B', 'C', 'D')
    + For Yes/No/Uncertain question, answer restricted in 'Yes' , 'No' , 'Uncertain'
    + For How many question, answer is a number.
    + For multiple-component question: answers separated by commas (e.g., 120, No).
  - `{idx}`:  A comma-separated list of the premise numbers (from `Premise #X`) that support the final answer.
  - `{explanation}`: Reasoning text **wrapped inside** <think> tags, written in natural language, clearly referring to the source premises (e.g., “From Premise 2, we know that…”).
"""

PROMPT_INPUT_INFO = """
Now, I will provide you with the premises and question:

## Domain knowledge:
<?premises>

## Question:
<?question>

That is all the information. Now it is your turn to respond.
"""