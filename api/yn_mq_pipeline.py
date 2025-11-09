import torch
import json
import re
import time
from typing import Dict, List, Any, Tuple, Optional

PROMPT_COT_SYS_YN ="""
You are a helpful assistant. 
Given a set of premises in natural language (NL) and a Yes/No/Uncertain question, 
Your task is to reason step by step based on the given premises in order to identify the most appropriate reasoning path to answer the question.

Provide a detailed explanation of your reasoning process, clearly referencing the premises used.
"""

PROMPT_COT_USER_STEP_1 = """
## Premises:
<?premises>

## Question: <?question> 

## Reasoning Steps:
""" 

PROMPT_COT_USER_STEP_2_YN = """
Based on your reasoning, what is the concise answer to the question? 
Answer strictly with only one of the following: 'Yes', 'No', or 'Uncertain'.
"""

PROMPT_COT_USER_STEP_3 = """
List the numeric indices (e.g., 1, 3, 7) of all the premises you used in your reasoning to reach the answer. Output only the numbers separated by commas.
"""

PROMPT_COT_USER_STEP_4 = """
Provide a concise step-by-step explanation for how you arrived at the answer using the identified premises.
"""

PROMPT_COT_USER_FINAL_YN = """
Now, assemble the final answer in the specified JSON format using the concise answer, the list of used premise indices, and the explanation you generated.

```json
{
  "answer": "<Yes | No | Uncertain>",
  "idx": [<List the index of premises, e.g., 1, 3>],
  "explanation": "<Your step-by-step logical reasoning>"
}
```
"""


QS_TYPE_YN = 'Yes/No/Uncertain'

class YnPipeline:
    def __init__(self, trace: bool = False):
        self.trace = trace

    def _generate_prompt(self, template: str, data: Dict[str, str]) -> str:
        prompt = template
        for key, val in data.items():
            prompt = prompt.replace(f'<?{key}>', val)
        return prompt

    def _create_sys_prompt(self) -> str:
        return self._generate_prompt(PROMPT_COT_SYS_YN, {})

    def _create_user_prompt(self, premises: Dict[str, str], question: str) -> str:
        premise_vals = ""
        for id, premise in premises.items():
            premise_vals += f"    - Premise {id}: {premise}\\n"
        
        data = {
            'premises': premise_vals,
            'question': question
        }
        return self._generate_prompt(PROMPT_COT_USER_STEP_1, data=data)

    def _create_user_step_2(self) -> str:
        """Creates the user prompt for step 2 (concise answer extraction)."""
        return self._generate_prompt(PROMPT_COT_USER_STEP_2_YN, {})

    def _create_user_step_3(self) -> str:
        """Creates the user prompt for step 3 (premise index identification)."""
        return self._generate_prompt(PROMPT_COT_USER_STEP_3, {})

    def _create_user_step_4(self) -> str:
        """Creates the user prompt for step 4 (explanation generation)."""
        return self._generate_prompt(PROMPT_COT_USER_STEP_4, {})

    def _create_user_final(self) -> str:
        """Creates the user prompt for the final step (structured output)."""
        return self._generate_prompt(PROMPT_COT_USER_FINAL_YN, {})

    def _generate_answer(self, tokenizer: Any, model: Any, messages: List[Dict[str, str]], max_tokens: int = 1000) -> str:
        """Generates a response from the model given the conversation history."""
        with torch.no_grad():
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize the text
            model_inputs = tokenizer([text], return_tensors="pt") 

            # Move each tensor in the dictionary to the model's device
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

            output = model.generate(
                **model_inputs, # Pass the dictionary contents as keyword arguments
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                temperature=0.7,
                top_p=0.9
            )

            generated_sequence = output.sequences[0]
            full_generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            
            # Need to get input_ids from the potentially modified model_inputs dictionary
            input_ids = model_inputs['input_ids'] 
            input_text_len = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
            return full_generated_text[input_text_len:].strip()

    def _parse_res(self, res_text: str) -> Optional[Dict[str, Any]]:
        """Parses the final JSON response from the model output."""
        if self.trace:
            print("\\nParsing final response...")
            
        res_text_cleaned = res_text.replace('\\n', '')
        matches = re.findall(r"```json\\s*([\\s\\S]*?)\\s*```", res_text_cleaned)
        
        parsed_result = None

        if matches:
            json_str = matches[0]
            try:
                parsed_result = json.loads(json_str)
                if self.trace:
                    print("Successfully parsed JSON response")
            except json.JSONDecodeError as e:
                if self.trace:
                    print(f"Error parsing JSON: {e}")
                    print("Raw JSON content attempted:")
                    print(json_str)
                parsed_result = self._extract_with_regex(res_text)
        else:
             if self.trace:
                print("No JSON block found in response, attempting regex extraction.")
             parsed_result = self._extract_with_regex(res_text)
        if parsed_result:
            if 'answer' not in parsed_result or not parsed_result['answer']:
                yes_no_match = re.search(r'\\b(yes|no|uncertain)\\b', res_text.lower())
                if yes_no_match:
                    parsed_result['answer'] = yes_no_match.group(1).capitalize()
                    if self.trace: print(f"Extracted fallback answer: {parsed_result['answer']}")
                else:
                    parsed_result['answer'] = 'Uncertain' 
            
            if 'idx' not in parsed_result or not isinstance(parsed_result.get('idx'), list):
                idx_matches = re.findall(r'premise\\s+(\\d+)', res_text.lower())
                if idx_matches:
                    parsed_result['idx'] = sorted(list(set([int(idx) for idx in idx_matches])))
                    if self.trace: print(f"Extracted fallback premise indices: {parsed_result['idx']}")
                else:
                     parsed_result['idx'] = []
            else:
                try:
                    parsed_result['idx'] = sorted(list(set([int(i) for i in parsed_result['idx'] if isinstance(i, (int, str)) and str(i).isdigit()])))
                except (ValueError, TypeError):
                     if self.trace: print("Error converting idx elements to int, defaulting to empty list.")
                     parsed_result['idx'] = []


            if 'explanation' not in parsed_result or not parsed_result['explanation']:
                parsed_result['explanation'] = res_text # Use raw response as fallback
                if self.trace: print("Using raw response as fallback explanation.")
        elif not parsed_result and res_text:
             if self.trace: print("Could not parse JSON or extract via regex, creating fallback structure.")
             # Create a fallback structure if everything fails but we have some text
             parsed_result = self._extract_with_regex(res_text) # Try one last time
             if not parsed_result.get('answer'): parsed_result['answer'] = 'Uncertain'
             if not parsed_result.get('idx'): parsed_result['idx'] = []
             if not parsed_result.get('explanation'): parsed_result['explanation'] = res_text

        return parsed_result


    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Attempts to extract answer, idx, and explanation using regex."""
        answer = ''
        idx = []
        explanation = text 

        answer_match = re.search(r'["\']?answer["\']?\s*[:=]\s*["\']?([^"\']+)["\']?', text, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            if answer.lower() in ['yes', 'no', 'uncertain']:
                answer = answer.capitalize()
        else:
             yes_no_match = re.search(r'\\b(yes|no|uncertain)\\b', text.lower())
             if yes_no_match:
                 answer = yes_no_match.group(1).capitalize()

        # Try extracting idx field specifically
        # Handles formats like: "idx": [1, 2], idx: [ 3 ,4 ], "idx"=[5,6], idx = [ 7 , 8 ]
        idx_match = re.search(r'["\']?idx["\']?\s*[:=]\s*\[([^\]]*)\]', text, re.IGNORECASE)
        if idx_match:
            idx_str = idx_match.group(1)
            try:
                # Extract numbers, handling potential extra spaces or quotes
                idx = sorted(list(set([int(i.strip().strip('"\'')) for i in idx_str.split(',') if i.strip().strip('"\'').isdigit()])))
            except (ValueError, TypeError):
                 if self.trace: print(f"Could not parse indices from regex match: {idx_str}")
                 idx = [] # Reset if parsing fails
        
        # If specific idx field extraction failed, try finding "Premise X" mentions
        if not idx:
            idx_matches = re.findall(r'premise\\s+(\\d+)', text.lower())
            if idx_matches:
                try:
                    idx = sorted(list(set([int(i) for i in idx_matches])))
                except ValueError:
                    idx = [] # Reset on error

        # Try extracting explanation field specifically
        explanation_match = re.search(r'["\']?explanation["\']?\s*[:=]\s*["\']?([^"\']+)["\']?', text, re.IGNORECASE)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        # else: keep default explanation (full text)

        return {
            "answer": answer,
            "idx": idx,
            "explanation": explanation
        }


    def run(self, premises: List[str], question: str, tokenizer: Any, model: Any) -> List[Dict[str, Any]]:
        """
        Executes the full Yes/No question answering pipeline.

        Args:
            premises (List[str]): A list of premise strings.
            question (str): The question to be answered.
            tokenizer: The tokenizer instance.
            model: The model instance.

        Returns:
            List[Dict[str, Any]]: A list containing one dictionary with 'answer', 'idx', and 'explanation'.
                                 Returns a list with an error dictionary if parsing fails.
        """
        pipeline_start_time = time.time()
        step_times: Dict[str, float] = {}

        # Format premises
        premise_dict = {str(i+1): premises[i] for i in range(len(premises))}

        # Prepare initial messages
        sys = self._create_sys_prompt()
        user = self._create_user_prompt(premises=premise_dict, question=question)
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ]

        # Step 1: Generate comprehensive answer
        if self.trace: print('Step 1: Generating comprehensive answer...')
        step_1_start_time = time.time()
        step_1_answer = self._generate_answer(tokenizer=tokenizer, model=model, messages=messages, max_tokens=800)
        step_times['step_1_generation'] = time.time() - step_1_start_time
        if self.trace: print(f"Step 1 took {step_times['step_1_generation']:.2f} seconds")

        # Step 2: Extract concise answer
        if self.trace: print('Step 2: Extracting concise answer...')
        step_2_start_time = time.time()
        messages.append({"role": "system", "content": step_1_answer}) # Use previous answer as context
        messages.append({"role": "user", "content": self._create_user_step_2()})
        step_2_answer = self._generate_answer(tokenizer=tokenizer, model=model, messages=messages, max_tokens=200)
        step_times['step_2_concise_answer'] = time.time() - step_2_start_time
        if self.trace: print(f"Step 2 took {step_times['step_2_concise_answer']:.2f} seconds")

        # Step 3: List index of used premises
        if self.trace: print('Step 3: Identifying used premises...')
        step_3_start_time = time.time()
        messages.append({"role": "system", "content": step_2_answer})
        messages.append({"role": "user", "content": self._create_user_step_3()})
        step_3_answer = self._generate_answer(tokenizer=tokenizer, model=model, messages=messages, max_tokens=100)
        step_times['step_3_premise_indices'] = time.time() - step_3_start_time
        if self.trace: print(f"Step 3 took {step_times['step_3_premise_indices']:.2f} seconds")

        # Step 4: Explanation
        if self.trace: print('Step 4: Generating explanation...')
        step_4_start_time = time.time()
        messages.append({"role": "system", "content": step_3_answer})
        messages.append({"role": "user", "content": self._create_user_step_4()})
        step_4_answer = self._generate_answer(tokenizer=tokenizer, model=model, messages=messages, max_tokens=300)
        step_times['step_4_explanation'] = time.time() - step_4_start_time
        if self.trace: print(f"Step 4 took {step_times['step_4_explanation']:.2f} seconds")

        # Step 5: Final structured answer
        if self.trace: print('Step 5: Generating final structured answer...')
        final_start_time = time.time()
        messages.append({"role": "system", "content": step_4_answer})
        messages.append({"role": "user", "content": self._create_user_final()})
        final_answer_raw = self._generate_answer(tokenizer=tokenizer, model=model, messages=messages, max_tokens=200)
        step_times['step_5_final_structure'] = time.time() - final_start_time
        if self.trace: print(f"Final step took {step_times['step_5_final_structure']:.2f} seconds")
        
        # Parse the final result
        parsing_start_time = time.time()
        parsed_result = self._parse_res(final_answer_raw)
        step_times['step_6_parsing'] = time.time() - parsing_start_time
        if self.trace: print(f"Parsing took {step_times['step_6_parsing']:.2f} seconds")

        total_time = time.time() - pipeline_start_time
        step_times['total_pipeline'] = total_time
        
        if self.trace:
            print(f"Total pipeline execution time: {total_time:.2f} seconds")
            print("Step-by-step timing:")
            for step, duration in step_times.items():
                print(f"  {step}: {duration:.2f} seconds")
            print(f"Final Parsed Result: {parsed_result}")


        # Ensure a dictionary is always returned if parsing was successful at any level
        if parsed_result is None:
             # If even regex extraction failed, return a minimal error structure
             if self.trace: print("ERROR: Final parsing failed completely.")
             error_result = { # Define the error dictionary
                 'answer': 'Error',
                 'idx': [],
                 'explanation': f"Failed to parse model output: {final_answer_raw}",
             }
             return [error_result] # Return the error dictionary in a list
        else:
             # Ensure essential keys exist, even if empty, before returning
             parsed_result.setdefault('answer', 'Uncertain')
             parsed_result.setdefault('idx', [])
             parsed_result.setdefault('explanation', 'No explanation available.')
             return [parsed_result] # Return the result dictionary in a list
