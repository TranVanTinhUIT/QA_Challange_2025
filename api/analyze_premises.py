import json
import torch
import logging
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_MIN_P = 0

ANALYZE_TOKEN_PER_PREMISE = 70
RETURN_TOKEN_PER_PREMISE = 50

PROMPT_SYS_GUILDLINE = """
You are a helpful assistant trained to analyze and understand the given domain knowledge. 
Given set of premises expressed as natural language, your task is to analyze all the given premises.

For each premise, extract:
 - Key entities, facts, event or definitions.
 - Logical conditions or rules (e.g., “if…then…”)
 - Represent the premise based on extracted information.
 (each premise response as "## Premises {index}: ...")
DO NOT include any inference.
Also include all extracted entities in response with the enclosed <entities></entities> tag, each separated by a comma.
"""

PROMPT_USER_INIT = """
## Premises:
<?premises>
"""

class Analyzer:
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

    def generate_answer(self, tokenizer, model, messages, max_new_tokens = None, enable_thinking: bool=False):
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
                max_new_tokens=model.config.max_position_embeddings, 
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

    def run(self, premises, question, tokenizer, model, disable_retrieval = False, trace=False):
        start_time = time.perf_counter()
        llm_call_times = 0
        
        guildline = PROMPT_SYS_GUILDLINE
        prompt_input = self.create_user_input(premises=premises, question=question)
        messages = [
            {"role": "system", "content": guildline},
            {"role": "user", "content": prompt_input}
        ]

        response = self.generate_answer(tokenizer=tokenizer, model=model, messages=messages, max_new_tokens= len(premises)*(ANALYZE_TOKEN_PER_PREMISE+ RETURN_TOKEN_PER_PREMISE))
        print('Response:', response)
        end_time = time.perf_counter()  

        return response


def get_model(model_name = "Qwen/Qwen2.5-32B-Instruct-AWQ"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model

def parse_number_or_zero(s):
    try:
        s = s.strip()
        return int(s)
    except (ValueError, AttributeError):
        return 0
    
def extract_analysis(text, original_premises):
    analysis_pattern = r"<entities>(.*?)</entities>"
    matches = re.findall(analysis_pattern, text, flags=re.DOTALL)

    entities = []
    for match in matches:
        items = [item.strip() for item in match.split(',')] 
        for item in items:
            if item not in entities:
                entities.append(item)

    clean_entities_text = re.sub(r'<entities>.*?</entities>', '', text, flags=re.DOTALL)

    pattern = r'## Premises (\d+):\s*(.*?)(?=## Premises \d+:|$)'
    matches = re.findall(pattern, clean_entities_text, flags=re.DOTALL)

    res_premises = original_premises
    for idx, content in matches:
        index = parse_number_or_zero(idx)
        new_content = content.strip()
        if index > 0 and index <= len(res_premises):
            res_premises[index - 1] = new_content

    return entities, res_premises

if __name__ == "__main__":
    analyzer = Analyzer()
    tokenizer, model = get_model() # Model

    ds_folder = os.path.dirname(os.path.abspath(__file__)) + '/../dataset'
    out_folder = os.path.dirname(os.path.abspath(__file__)) + '/data'

    with open(ds_folder + '/hm_train.json', "r", encoding="utf-8") as f:
        hm_train = json.load(f)
    
    test_ds = hm_train
    print(len(test_ds))

    results = []

    out_file = out_folder + f"/preprocessed_train_v1_analyze.json"
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    
    for index, item in enumerate(test_ds):
        if index < len(results):
            continue
        print('Test item ', index)
        if index %3 == 0 :
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        premises = item['premises-NL']
        question = item['question']
        item['premises-NL-old'] = list(premises)
        start_time = time.perf_counter()
        result = analyzer.run(premises=premises, question=question, tokenizer=tokenizer, model=model, disable_retrieval=False, trace=True)
        entities, rewritten_premises = extract_analysis(result, premises)

        endtime_time = time.perf_counter()
        
        
        item['premises-NL'] = rewritten_premises
        item['entities'] = entities

        results.append(item)
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
   