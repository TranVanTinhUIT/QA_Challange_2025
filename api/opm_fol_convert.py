
import torch
import re
import json
from nltk.sem.logic import *

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

PROMPT_GUILDLINE_CONVERT_PREMISES = """
You are a reasoning assistant trained to perform convert natural language premises to first order logic by integrate Object-Process Methodology (OPM) in ISO 19450:2015 standard. 

Given a set of premises in natural language, your tasks is to follow these steps:
   1. Modeling the given premises follow Object-Process methodology to obtain predicates set for first-order logic. The predicates set includes Objects, States of OPM.
   2. Based on the set of obtaining predicates, convert each the given premise to its first order logic representation.

First-order logic representations must follow conventions:
 -  Variable: using single lowercase letter. Ex: x, y, z
 -  Logical Connectives:  use standard symbols as follow:
   + `¬` for NOT.
   +  `∧` for AND
   +  `∨` for OR
   +  `→` for IMPLIES.
   +  `↔` for IFF
 - Quantifiers:  use ∀ (for all) and ∃ (there exists) correctly.

Response in following JSON  format:
```json
{
  "Predicates": [{"predicate": "<predicate>", "description": "<description>"}],
  "Premises_FOL": [{ "index": "<index>" , "fol":  "<first order logic>"}]
}

Where:
  - `Predicates` is list of predicates obtain in step 1 when modeling OPM. Each item in list includes:
       + `predicate`: name of the predicate according to the PascalCase convention (Capitalize the first letter of every word).
       + `description`: concise description about what the predicate represent for. Ex: Represent for ABC.
  - `Premises_FOL` is list first order logic representation of All the given premise. Each item in the list includes:
       + `index` is the premise index with index is X of premise #X.
       + `first order logic` is the corresponding first-order logic representation of the premise.

IMPORTANT: All of the premises including both assumptions and logical rules must be converted to it first order logic representation to ensure consistency.
```

"""

PROMPT_CONVERT_PREMISES_INPUT = """
## Premises:
<?premises>

"""

PROMPT_GUILDLINE_CONVERT_YN_QUESTION = """
You are a reasoning assistant trained to perform convert natural language premises to first order logic by integrate Object-Process Methodology (OPM) in ISO 19450:2015 standard. 

Given a set of premises in natural language and a Yes/No question, your tasks is to follow these steps:
   1. Modeling the given premises follow Object-Process methodology to obtain predicates set for first-order logic. The predicates set includes Objects, States of OPM.
   2. Based on the set of obtaining predicates, convert each the given premise to its first order logic representation. Every premise must be converted. Skipping any premise is not allowed to ensure logical consistency.
   3. Based on the set of obtaining predicates, convert the conclusion in the given question to its first order logic representation.

First-order logic representations must follow conventions:
 -  Variable: using single lowercase letter. Ex: x, y, z
 -  Logical Connectives:  use standard symbols as follow:
   + `¬` for NOT.
   +  `∧` for AND
   +  `∨` for OR
   +  `→` for IMPLIES.
   +  `↔` for IFF
 - Quantifiers:  use ∀ (for all) and ∃ (there exists) correctly.

Response in following JSON  format:
```json
{
  "Predicates": [{"predicate": "<predicate>", "description": "<description>"}],
  "Premises_FOL": [{ "index": "<index>" , "fol":  "<first order logic>"}],
   "Conclusion_FOL": ""
}

Where:
  - `Predicates` is list of predicates obtain in step 1 when modeling OPM. Each item in list includes:
       + `predicate`: name of the predicate according to the PascalCase convention (Capitalize the first letter of every word).
       + `description`: concise description about what the predicate represent for. Ex: Represent for ABC.

  - `Premises_FOL` is list corresponding first order logic representation of each the given premises. Each item in the list includes:
       + `index` is the premise index with index is X of premise #X.
       + `first order logic` is the corresponding first-order logic representation of the premise.
  IMPORTANT: All of the premises including both assumptions and logical rules must be converted to it first order logic representation to ensure consistency.

  -  `Conclusion_FOL` is first-order logic representation of the conclusion in the question.

```

"""

PROMPT_CONVERT_QUESTION_INPUT = """
## Premises:
   <?premises>

## Question: <?question>

"""

PROMPT_CORRECT_RESPONSE_OLD = """
Some premises are missing their first order logic representation: <?missing_premises>. Re-add their representation into JSON response. 
Respond in a single JSON format by merging all.
"""

PROMPT_CORRECT_RESPONSE = """
There are some issues in your response:
    1. Some premises are missing their first order logic representation: <?missing_premises>. Re-add their representation into JSON response. 
    2. Some premises are missing quantified variables: <?missing_quantified_preimises>.  Add the corresponding quantified variables to correct the first order representation of the premise.
Resolve these issue and respond all in a single JSON format.
"""

PROMPT_GUILDLINE_CONVERT_CHOICE_QUESTION = """
You are a reasoning assistant trained to perform convert natural language premises to first order logic by integrate Object-Process Methodology (OPM) in ISO 19450:2015 standard. 

Given a set of premises in natural language and a multiple choice question, your tasks is to follow these steps:
   1. Modeling the given premises follow Object-Process methodology to obtain predicates set for first-order logic. The predicates set includes Objects, States of OPM.
   2. Based on the set of obtaining predicates, convert each the given premise to its first order logic representation. Every premise must be converted. Skipping any premise is not allowed to ensure logical consistency.
   3. Based on the set of obtaining predicates, convert the extra assumptions and each options in the given question to their first order logic representation.

First-order logic representations must follow conventions:
 -  Variable: using single lowercase letter. Ex: x, y, z
 -  Logical Connectives:  use standard symbols as follow:
   + `¬` for NOT.
   +  `∧` for AND
   +  `∨` for OR
   +  `→` for IMPLIES.
   +  `↔` for IFF
 - Quantifiers:  use ∀ (for all) and ∃ (there exists) correctly.

Response in following JSON  format:
```json
{
  "Predicates": [{"predicate": "<predicate>", "description": "<description>"}],
  "Premises_FOL": [{ "index": "<index>" , "fol":  "<premise first order logic>"}],
  "Assumtions_FOL": [<List the extra assumptions in the question>],
   "Options_FOL": [{"option": "<letter of option>", "fol": <option first order logic>}]
}

Where:
  - `Predicates` is list of predicates obtain in step 1 when modeling OPM. Each item in list includes:
       + `predicate`: name of the predicate according to the PascalCase convention (Capitalize the first letter of every word).
       + `description`: concise description about what the predicate represent for. Ex: Represent for ABC.

  - `Premises_FOL` is list corresponding first order logic representation of each the given premises. Each item in the list includes:
       + `index` is the premise index with index is X of premise #X.
       + `premise first order logic` is the corresponding first-order logic representation of the premise.
  IMPORTANT: All of the premises including both assumptions and logical rules must be converted to it first order logic representation to ensure consistency.

  -  `Assumtions_FOL` is list first-order logic representation of the extra assumptions in the question. If there is no extra assumptions, set empty array for it.
  -  `Options_FOL` is  is list corresponding first order logic representation of each option. Each item in the list includes:
       + `letter of option` is the letter of the option such as A, B, C, D. 
       + `option first order logic` is the corresponding first-order logic representation of the option.

```

"""

PROMPT_GUILDLINE_CONVERT_HM_QUESTION = """

"""

class OPMFolConvert:
    """
    Define class for convert premises to FOL by integrate Object-process methodology (OPM)
    """
    def __init__(self):
        pass
    
    def fol_to_expression_format(self, fol: str): 
        replacements = {
            '∀': ' all ',
            '∃': ' exists ',
            '¬': '-',
            '∧': '&',
            '⊕': '!=',
            '∨': '|',
            '→': '->',
            '↔': '<->',
            'FORALL': ' all ',
            'EXISTS': ' exists ',
            'NOT': ' - ',
            'AND': '&',
            'XOR': '!=',
            'OR': '|',
            'THEN': '->',
            'IFF': '<->',
        }
        for symbol, nltk in replacements.items():
            fol = re.sub(re.escape(symbol), nltk, fol)
        return fol

    def is_missing_quatified_variable(self, parsed_premise_fol):
        try:
            nltk_exp = self.fol_to_expression_format(parsed_premise_fol)
            exp = Expression.fromstring(nltk_exp)
            free_variables = exp.free()
            return len(free_variables) > 0
        except:
            return False

    def create_convert_premises_input(self, premises):
        input = PROMPT_CONVERT_PREMISES_INPUT

        premise_str = ''
        for i in range(len(premises)):
            premise_str += f'    - Premise {i+1}: {premises[i]}.\n'

        input = input.replace('<?premises>', premise_str)

        return input
    
    def create_convert_question_input(self, premises, question):
        input = PROMPT_CONVERT_QUESTION_INPUT

        premises_str = ''
        for i in range(len(premises)):
            premises_str += f'     - Premise #{i+1}: {premises[i]}\n'

        input = input.replace('<?premises>', premises_str)
                
        # question
        input = input.replace('<?question>', question)
        
        return input
    
    def create_convert_choice_question_input(self, premises, question):
        input = PROMPT_CONVERT_QUESTION_INPUT

        premises_str = ''
        for i in range(len(premises)):
            premises_str += f'     - Premise #{i+1}: {premises[i]}\n'

        input = input.replace('<?premises>', premises_str)
                
        # question
        input = input.replace('<?question>', question)
        
        return input
    
    def create_correct_response(self, missing_indices, missing_quantifications):
        input = PROMPT_CORRECT_RESPONSE

        # Premises which is missing FOL
        missing_premises_arr = [f'Premise #{index}' for index in missing_indices]

        missing_premises_str = ", ".join(missing_premises_arr)
        input = input.replace('<?missing_premises>', missing_premises_str)

        # Premises which is missing quantified variables
        missing_quantified_arr = [f'Premise #{index}' for index in missing_quantifications]
        missing_quantified_str = ", ".join(missing_quantified_arr)
        input = input.replace('<?missing_quantified_preimises>', missing_quantified_str)

        return input

    def convert_premises(self, premises, tokenizer, model):
        guildline = PROMPT_GUILDLINE_CONVERT_PREMISES

        input = self.create_convert_premises_input(premises=premises)
        messages = [
            {"role": "system", "content": guildline},
            {"role": "user", "content": input}
        ]

        output = self.generate(tokenizer=tokenizer, model=model, messages=messages, max_new_tokens=2000)

        return self.parse_response(output)

    def convert_yn_question(self, premises, yn_question, tokenizer, model):
        guildline = PROMPT_GUILDLINE_CONVERT_YN_QUESTION

        input = self.create_convert_question_input(premises=premises, question=yn_question)
        messages = [
            {"role": "system", "content": guildline},
            {"role": "user", "content": input}
        ]

        print('MESSAGES:', messages)

        # Round 1: Convert Premises and Conclusion into their First-Order logic representation.
        output_round_1 = self.generate(tokenizer=tokenizer, model=model, messages=messages, max_new_tokens=1500)
        print('OUTPUT ROUND 1:', output_round_1)
        response_round_1 = self.parse_response(output_round_1)
        if response_round_1 is None:
            return None

        # check missing converting premises
        missing_premise_indices = []
        premises_FOL = response_round_1.get('Premises_FOL', [])
        premise_indices = [int(premise_FOL['index']) for premise_FOL in premises_FOL]
        for i in range(len(premises)):
            index = i + 1
            if index not in premise_indices:
                missing_premise_indices.append(index)
        # check incorrect rules (missing quantified variable)
        missing_quantified_indices = []
        for premise_FOL in premises_FOL:
            index = int(premise_FOL['index'])
            parsed_fol = premise_FOL['fol']
            if self.is_missing_quatified_variable(parsed_fol):
                missing_quantified_indices.append(index)
            
        if len(missing_premise_indices) == 0 and len(missing_quantified_indices) == 0:
            return response_round_1

        messages.append({"role": "system", "content": output_round_1})
        messages.append({"role": "system", "content": self.create_correct_response(missing_indices=missing_premise_indices, missing_quantifications=missing_quantified_indices)})

        # Round 2: Correct the response by add missing premises
        output_round_2 = self.generate(tokenizer=tokenizer, model=model, messages=messages, max_new_tokens=1500)
        print('OUTPUT ROUND 2:', output_round_2)
        return self.parse_response(output_round_2)

    def convert_choice_question(self, premises, choice_question, tokenizer, model):
        guildline = PROMPT_GUILDLINE_CONVERT_CHOICE_QUESTION

        input = self.create_convert_question_input(premises=premises, question=choice_question)
        messages = [
            {"role": "system", "content": guildline},
            {"role": "user", "content": input}
        ]

        print('MESSAGES:', messages)

        # Round 1: Convert Premises and Conclusion into their First-Order logic representation.
        output_round_1 = self.generate(tokenizer=tokenizer, model=model, messages=messages, max_new_tokens=1500)
        print('OUTPUT ROUND 1:', output_round_1)
        response_round_1 = self.parse_response(output_round_1)
        if response_round_1 is None:
            return None

        # Correct the response by check missing converting premises
        missing_premise_indices = []
        premises_FOL = response_round_1.get('Premises_FOL', [])
        premise_indices = [int(premise_FOL['index']) for premise_FOL in premises_FOL]
        for i in range(len(premises)):
            index = i + 1
            if index not in premise_indices:
                missing_premise_indices.append(index)

        if len(missing_premise_indices) == 0:
            return response_round_1

        messages.append({"role": "system", "content": output_round_1})
        messages.append({"role": "system", "content": self.create_correct_response(missing_indices=missing_premise_indices)})

        # Round 2: Correct the response by add missing premises
        output_round_2 = self.generate(tokenizer=tokenizer, model=model, messages=messages, max_new_tokens=1500)
        print('OUTPUT ROUND 2:', output_round_2)
        return self.parse_response(output_round_2)

    def convert_hm_question(self, premises, hm_question, tokenizer, model):
        pass

    def parse_response(self, llm_response):
        """
        Parse response to json in standard json format. If cannot parse, a fallback will be used instead. (see `parse_response_round_2`)
        """
        llm_response = llm_response.replace('\n', '')
        matchs = re.findall(r"```json\s*([\s\S]*?)\s*```", llm_response)
        if matchs:
            return json.loads(matchs[0])

        return None

    def generate(self, tokenizer, model, messages, max_new_tokens = 1000):
        with torch.no_grad():
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            input_ids = model_inputs.input_ids

            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True
            )

        # Decode output (including both prompt and generated text)
        generated_sequence = output.sequences[0]
        full_generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        
        # Get only generated text (remove prompt)
        input_text_len = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
        return full_generated_text[input_text_len:].strip()
    

# def get_model(model_name = "Qwen/Qwen2.5-7B-Instruct", bnb_config = BitsAndBytesConfig(load_in_8bit=True,llm_int8_threshold=6.0,)):
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=bnb_config,
#         device_map="auto",
#         trust_remote_code=True,
#     )
#     return tokenizer, model

# if __name__ == '__main__':

#     tokenizer, model = get_model(model_name="Qwen/Qwen2.5-7B-Instruct") # Model

#     premises = [
#       "If a faculty member has completed training, they can teach undergraduate courses.",
#       "If a faculty member can teach undergraduate courses and holds a PhD, they can supervise graduate students.",
#       "If a faculty member can supervise graduate students and has at least 3 publications, they can serve on curriculum committees.",
#       "If a faculty member can serve on curriculum committees and has a positive teaching evaluation, they can propose new courses.",
#       "Professor John has completed pedagogical training.",
#       "Professor John holds a PhD.",
#       "Professor John has published at least 3 academic papers.",
#       "Professor John has received a positive teaching evaluation."
#     ]

#     yn_question = "Does the logical chain demonstrate that Professor John meets all requirements for collaborative research projects?"

#     converter = OPMFolConvert()
#     start_time = time.perf_counter()
#     output_premises = converter.convert_yn_question(premises=premises, yn_question=yn_question, tokenizer=tokenizer, model=model)
#     end_time = time.perf_counter()
#     print('Execution_time: ', end_time - start_time)
#     # print('OUTPUT:', output)
#     if output_premises:
#         predicates = output_premises.get('Predicates', [])
#         print('Predicates:')
#         for predicate in predicates:
#             predicate_name = predicate.get('predicate', '')
#             description = predicate.get('description', '')
#             print(f'   {predicate_name} ::: {description}')

#         premises_fol = output_premises.get('Premises_FOL', [])
#         print('First order logic:')
#         fols = []
#         for i in range(len(premises_fol)):
#             index = premises_fol[i].get('index', '')
#             logic = premises_fol[i].get('fol', '')
#             fols.append(logic)
#             print(f'   {index}. {logic} ::: {premises[i]}')

#         conclusion = output_premises.get('Conclusion_FOL', '')
#         print('Conclusion: ', conclusion)
#     else:
#         print('Cannot convert premises')
    