import json
import re
import time
import torch

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_MIN_P = 0

class ChoicePipeline3:
    """
    Choice pipeline integrating OPM with only one times call LLM
    """
    def __init__(self):
        pass

    def generate_prompt(self, template, dict):
        prompt = template

        for key, val in dict.items():
            prompt = prompt.replace(f'<?{key}>', val)
        
        return prompt
    
    def create_user_input(self, premises, question):
        """
        create prompt input premises and question from prompt template
        """
        
        dict = {}
        
        premise_list = [{ 'idx': (i+1), 'premise': premises[i] } for i in range(len(premises))]

        # <?premises>
        premise_vals = ""

        for premise_item in premise_list:
            premise_idx = premise_item['idx']
            premise = premise_item['premise']

            premise_vals += f"    - Premise {premise_idx}: {premise} \n"
        dict['premises'] = premise_vals

        # <?question> 
        dict['question'] = question

        return self.generate_prompt(PROMPT_USER_INIT, dict=dict)

    def generate_answer(self, tokenizer, model, messages, max_new_tokens = 1000, enable_thinking: bool=False):
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
        start_time = time.perf_counter()
        llm_call_times = 0
        
        guildline = PROMPT_SYS_GUILDLINE

        prompt_input = self.create_user_input(premises=premises, question=question)
        messages = [
            {"role": "system", "content": guildline},
            {"role": "user", "content": prompt_input}
        ]
        if trace:
            print(guildline)
            print(prompt_input)

        finalize_response = self.generate_answer(tokenizer=tokenizer, model=model, messages=messages, max_new_tokens= 2000, enable_thinking=enable_thinking)
        if trace:
            print(finalize_response)
        llm_call_times +=1
        step1_completed_time = time.perf_counter()
        step1_cost = step1_completed_time - start_time

        final_answer = { 'question': question, 'answer': '', 'idx': [], 'explanation': '', 'res': finalize_response , 'error': '', 'manual': False }
        try:
            parsed_res = self.parse_response(finalize_response)
            final_answer['answer'] = parsed_res['answer']
            final_answer['idx'] = parsed_res['idx']
            final_answer['explanation'] = parsed_res['explanation']
        except Exception as e:
            if trace:
                print('   Cannot parse response: ', str(e))
            final_answer['error'] = str(e)
            manual_parsed_answer =self.parse_response_round_2(final_answer)
            final_answer['answer'] = manual_parsed_answer['answer']
            final_answer['idx'] = manual_parsed_answer['idx']
            final_answer['explanation'] = manual_parsed_answer['explanation']
            final_answer['manual'] = True

        final_answer['time'] = step1_cost
        return final_answer
    

PROMPT_USER_INIT = """
## Premises:
<?premises>

## Question: <?question> 

"""

PROMPT_SYS_GUILDLINE = """
You are a reasoning assistant trained to modeling the given natural language premises using Object Process methodology (OPM) according to ISO 19450:2015 standard and logical inference on OPM to answer multiple-choice questions.
Given a set of premises expressed in natural language, a multiple-choice question and a list of answer options. Your task is to follow these steps:
  1. **Modeling**: Identify and define relevant objects and their states using OPM.
  2. **Representation*: Map each premise to components of the OPM model.
  3. **Inference**: Perform logical inference on premise presentation to evaluate each answer option:
    - Apply logical reasoning step by step, based on representation of the premises.
    - Evaluate whether the option logically follows or conflicts with the information provided.
    - Provide a detailed explanation of your reasoning, explicitly referencing relevant premises in each step.
  3. Conclusion: After evaluating all options, determine which one is best supported by the premises and reasoning. Clearly state your final answer.
  4. Finalize your response in the following format:
    ```xml
    <response>
      <answer>{answer}</answer>
      <idx>{idx}</idx>
      <explanation>{explanation}</explanation>
    </response>
    ```
    Where:
    - `{answer}` is  the chosen option’s letter. eg 'A', 'B', 'C', 'D'
    - `{explanation}` is a detailed textual explanation. Typically, includes three components:
        + A clear justification for the correct answer with referenced premises.
        + Briefly explain why each of the other options is incorrect, showing how they conflict with or are unsupported by the premises.
        + A concluding statement confirming your choice.
    - `{idx}` is list the numerical indexes of the premises (from `Premise #X`) that support the chosen answer.
"""