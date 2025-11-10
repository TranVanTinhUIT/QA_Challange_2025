import json
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_MIN_P = 0

class ChoicePipelineNoCoT:
    """
    Choice pipeline without CoT
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
You are a reasoning assistant trained to answer the question based on the given domain knowledge.
Your task is to answer the question using the given domain knowledge expressed in set of natural language premises.

Put your response to following format:
```
<response>
    <answer>{answer}</answer>
    <explanation>{explanation}</explanation>
    <idx>{idx}</idx>
</response>
```
Field description:
  - `{answer}`: The final concise answer, answer only is your choice letter. (e.g. 'A', 'B', 'C', 'D')
  - `{explanation}`: Your reasoning text written in natural language, clearly referring to the source premises (e.g., “From Premise 2, we know that…”).
  - `{idx}`: A comma-separated list of the premise numbers (from `Premise #X`) that support the final answer.
"""

def get_model(model_name = "Qwen/Qwen2.5-32B-Instruct-AWQ"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model

if __name__ == "__main__":
    pipeline = ChoicePipelineNoCoT() 
    tokenizer, model = get_model() # Model

    ds_folder = os.path.dirname(os.path.abspath(__file__)) + '/../dataset'
    out_folder = os.path.dirname(os.path.abspath(__file__)) + '/../out'

    with open(ds_folder + '/choice_test.json', "r", encoding="utf-8") as f:
        choice_test = json.load(f)
    
    test_ds = choice_test
    print(len(test_ds))

    results = []

    out_file = out_folder + f"/output_choice_no_CoT.json"
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    
    for index, item in enumerate(test_ds):
        if index < len(results):
            continue
        print('Test item ', index)
        if index %50 == 0 :
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        premises = item['premises-NL']
        question = item['question']
        
        choice_result = pipeline.run(premises=premises, question=question, tokenizer=tokenizer, model=model, trace=True)

        result = {
            'premises': premises,
            'question': question
        }
        
        result['ref_answer'] = item['answer'],
        result['ref_index'] = item['idx']
        result['ref_explanation'] = item['explanation']
        result['pred_answer'] = choice_result['answer']
        result['pred_idx'] = choice_result['idx']
        result['pred_explanation'] = choice_result['explanation']
        results.append(result)
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)