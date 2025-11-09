import torch
import json
import os
from typing import List, Dict, Any
import re
import time
from datetime import datetime

'''
PROMPT_COT_SYS_YN ="""
You are a helpful assistant. Given premises and a Yes/No/Uncertain question,
reason step-by-step using only the provided premises to determine the answer.
Focus on logical deduction and clearly state which premises support each step.
If the premises do not contain sufficient information to answer the question with certainty, 
your answer should be 'Uncertain' with an empty list of premise indices.
"""
'''

PROMPT_COT_SYS_YN = """
You are a helpful assistant. Given premises and a Yes/No/Uncertain question, reason step-by-step using ONLY the minimal set of premises necessary to answer the question. Focus on logical deduction and clearly state which premises support each step. Avoid including premises that are not directly required to derive the answer. If the premises do not contain sufficient information to answer the question with certainty, your answer must be 'Uncertain' with an empty list of premise indices.

**Important**: All final responses must be in valid JSON format with the structure:
{
  "answer": "Yes|No|Uncertain",
  "idx": [int],
  "explanation": "string"
}
Example:
{
  "answer": "Yes",
  "idx": [1, 2],
  "explanation": "Premise 1 states 'A implies B' and Premise 2 states 'B implies C'. Since A is true, C is true."
}
If the answer is 'Uncertain', idx must be [].
"""

PROMPT_COT_USER_STEP_1 = """
## Premises:
<?premises>

## Question: <?question>

## Reasoning Steps Towards Answer:
(Provide a focused, step-by-step deduction using the premises to answer the question.)
"""

PROMPT_COT_USER_STEP_2_YN = """
Based on the reasoning, what is the concise answer? Strictly 'Yes', 'No', or 'Uncertain'.
Remember, if the premises do not provide sufficient information to determine a definitive 'Yes' or 'No' answer, you must respond with 'Uncertain'.
"""

PROMPT_COT_USER_STEP_3_EXPLANATION = """
Based on the initial reasoning and the concise answer ('<?concise_answer>'), provide a step-by-step explanation for reaching the answer.
**Follow these strict instructions:**
1.  Begin each step with "Step X:".
2.  Explicitly state the premise number(s) used in each step (e.g., "Using Premise 1...").
3.  Briefly quote or paraphrase the relevant part of the premise(s) used in that step.
4.  Explain the logical deduction made in that step based *only* on the cited premises.
5.  Do not introduce any external knowledge or information not present in the original premises.
6.  Conclude with a final step that states the answer derived from the preceding steps.
7.  Keep the explanation concise and directly focused on answering the question.
8.  If the answer is 'Uncertain', explain why the premises are insufficient to determine a definitive answer.

Explanation:
"""

PROMPT_COT_USER_STEP_4_IDX = """
Review the step-by-step explanation provided:
"<?explanation>"

List the numeric indices of **all and only** the original premises explicitly mentioned or cited (e.g., "Premise 1", "Using Premise 3 and 5") in the explanation.
**Output Format:**
-   Provide *only* the numbers, separated by commas (e.g., 1,3,5).
-   Do not include any other text, labels, or explanations.
-   If no premises were cited, output nothing.
-   If the answer is 'Uncertain' because the premises are insufficient, leave this empty.
"""

PROMPT_COT_USER_FINAL_YN = """
Based on the following information:
- Concise Answer: '<?concise_answer>'
- Identified Premise Indices: <?idx_list_str> # String representation, e.g., "1,3,5" or ""
- Step-by-Step Explanation: "<?explanation>"

Construct the final response in EXACTLY the following JSON format. Ensure:
- No leading or trailing commas.
- No extra commas between fields or within arrays.
- Proper escaping of quotes and special characters in the explanation.
- Output ONLY the JSON object, without any additional text, code block markers (e.g., ```json), or comments.

{
  "answer": "<Yes|No|Uncertain>",
  "idx": [<List of premise indices as integers>],
  "explanation": "<Detailed explanation with proper escaping>"
}

Example:
{
  "answer": "Uncertain",
  "idx": [],
  "explanation": "The premises do not provide sufficient information to confirm a causal chain."
}

If the answer is 'Uncertain', set idx to [] and explain why the premises are insufficient. Ensure the explanation is concise and properly escaped for JSON. Use the provided concise answer and indices as a guide, but validate that idx is empty for 'Uncertain'.
"""

QS_TYPE_YN = 'Yes/No/Uncertain'

def generate_prompt(template, data_dict):
    prompt = template
    for key, val in data_dict.items():
        target = f'<?{key}>'
        if target in prompt:
            prompt = prompt.replace(target, str(val))
        else:
            pass
    return prompt

def create_sys_prompt(qs_type):
    if qs_type == QS_TYPE_YN:
        return generate_prompt(PROMPT_COT_SYS_YN, data_dict={})
    else:
        print(f"Warning: Unexpected question type '{qs_type}' in create_sys_prompt")
        return ""

def create_user_prompt(premises: List[str], question: str):
    data_dict = {}
    premise_vals = ""
    for i, premise in enumerate(premises, 1):
        premise_vals += f"    - Premise {i}: {premise}\n"
    data_dict['premises'] = premise_vals
    data_dict['question'] = question
    return generate_prompt(PROMPT_COT_USER_STEP_1, data_dict=data_dict)

def create_user_step_2(qs_type):
    if qs_type == QS_TYPE_YN:
        return generate_prompt(PROMPT_COT_USER_STEP_2_YN, {})
    else:
        print(f"Warning: Unexpected question type '{qs_type}' in create_user_step_2")
        return ""

def create_user_step_3_explanation(concise_answer):
    return generate_prompt(PROMPT_COT_USER_STEP_3_EXPLANATION, {"concise_answer": concise_answer})

def create_user_step_4_idx(explanation):
    return generate_prompt(PROMPT_COT_USER_STEP_4_IDX, {"explanation": explanation})

def create_user_final(concise_answer, idx_list_str, explanation):
    try:
        if idx_list_str and idx_list_str.strip():
            parsed_idx = [int(i.strip()) for i in idx_list_str.split(',') if i.strip().isdigit()]
            idx_list_json = json.dumps(sorted(list(set(parsed_idx))))
        else:
            idx_list_json = "[]"
    except Exception as e:
        print(f"Warning: Error converting index list string '{idx_list_str}' to JSON list: {e}")
        idx_list_json = "[]"

    cleaned_explanation = explanation.strip()
    cleaned_explanation_escaped = json.dumps(cleaned_explanation)[1:-1]

    return generate_prompt(PROMPT_COT_USER_FINAL_YN, {
        "concise_answer": concise_answer,
        "idx_list_str": idx_list_str,
        "idx_list_json": idx_list_json,
        "explanation": cleaned_explanation_escaped
    })

def generate_pipeline(premises: List[str], question: str, tokenizer, model, qs_type = QS_TYPE_YN, trace=False):
    pipeline_start_time = time.time()
    step_times = {}

    print(f"Using fixed question type: {qs_type}")
    if qs_type != QS_TYPE_YN:
        print(f"Error: This pipeline is configured only for '{QS_TYPE_YN}' questions.")
        return None, [], {'error': f"Unsupported question type: {qs_type}"}

    sys = create_sys_prompt(qs_type)
    user = create_user_prompt(premises=premises, question=question)
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user}
    ]

    if trace:
        print('Step 1: Generating initial reasoning...')

    step_1_start_time = time.time()
    step_1_reasoning = generate_answer(tokenizer=tokenizer, model=model, messages=messages, max_tokens=750)
    step_times['step_1_reasoning'] = time.time() - step_1_start_time

    if trace:
        print(f"Step 1 (Reasoning) took {step_times['step_1_reasoning']:.2f} seconds")

    if trace:
        print('Step 2: Extracting concise answer...')

    step_2_start_time = time.time()
    messages_step2 = messages + [{"role": "system", "content": step_1_reasoning}, {"role": "user", "content": create_user_step_2(qs_type)}]
    step_2_concise_answer_raw = generate_answer(tokenizer=tokenizer, model=model, messages=messages_step2, max_tokens=30)
    match = re.search(r'\b(Yes|No|Uncertain)\b', step_2_concise_answer_raw, re.IGNORECASE)
    step_2_concise_answer = match.group(1).capitalize() if match else "Uncertain"
    step_times['step_2_concise_answer'] = time.time() - step_2_start_time

    if trace:
        print(f"Step 2 (Concise Answer) took {step_times['step_2_concise_answer']:.2f} seconds")
        print(f"Step 2 Concise Answer Output: {step_2_concise_answer}")

    if trace:
        print('Step 3: Generating concise explanation...')

    step_3_start_time = time.time()
    messages_step3 = messages + [
        {"role": "system", "content": step_1_reasoning},
        {"role": "system", "content": f"Concise Answer: {step_2_concise_answer}"},
        {"role": "user", "content": create_user_step_3_explanation(step_2_concise_answer)}
    ]
    step_3_explanation = generate_answer(tokenizer=tokenizer, model=model, messages=messages_step3, max_tokens=600)
    step_times['step_3_explanation'] = time.time() - step_3_start_time

    if trace:
        print(f"Step 3 (Explanation) took {step_times['step_3_explanation']:.2f} seconds")

    if trace:
        print('Step 4: Extracting premise indices...')

    step_4_start_time = time.time()
    messages_step4 = messages_step3 + [
        {"role": "system", "content": step_3_explanation},
        {"role": "user", "content": create_user_step_4_idx(step_3_explanation)}
    ]
    step_4_indices_str = generate_answer(tokenizer=tokenizer, model=model, messages=messages_step4, max_tokens=50)
    step_4_indices_str = ",".join(re.findall(r'\b(\d+)\b', step_4_indices_str))
    step_times['step_4_indices'] = time.time() - step_4_start_time

    if trace:
        print(f"Step 4 (Indices) took {step_times['step_4_indices']:.2f} seconds")
        print(f"Step 4 Indices Output: '{step_4_indices_str}'")

    if trace:
        print('Step 5: Generating final JSON output...')

    final_start_time = time.time()
    messages_step5 = messages_step4 + [
        {"role": "system", "content": f"Context for JSON Generation:\nConcise Answer: {step_2_concise_answer}\nIndices String: {step_4_indices_str}\nExplanation:\n{step_3_explanation}"},
        {"role": "user", "content": create_user_final(step_2_concise_answer, step_4_indices_str, step_3_explanation)}
    ]
    final_output_str = generate_answer(tokenizer=tokenizer, model=model, messages=messages_step5, max_tokens=700)
    step_times['final_json_generation'] = time.time() - final_start_time

    if trace:
        print(f"Step 5 (Final JSON Generation) took {step_times['final_json_generation']:.2f} seconds")

    total_time = time.time() - pipeline_start_time
    step_times['total_pipeline'] = total_time

    if trace:
        print(f"Total pipeline execution time: {total_time:.2f} seconds")
        print("Step-by-step timing:")
        for step, duration in step_times.items():
            print(f"  {step}: {duration:.2f} seconds")

    return final_output_str, step_times

def generate_answer(tokenizer, model, messages, max_tokens=1000):
    with torch.no_grad():
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            print("Error: 'messages' must be a list of dictionaries.")
            return json.dumps({
                "answer": "Error",
                "idx": [],
                "explanation": "Invalid message format"
            })

        valid_messages = [m for m in messages if m.get("content")]
        if not valid_messages:
            print("Error: No valid messages to process.")
            return json.dumps({
                "answer": "Error",
                "idx": [],
                "explanation": "No valid messages"
            })

        try:
            formatted_messages = []
            for msg in valid_messages:
                if msg["role"] == "system":
                    formatted_messages.append(f"<|im_start|>system\n{msg['content']}<|im_end|>")
                elif msg["role"] == "user":
                    formatted_messages.append(f"<|im_start|>user\n{msg['content']}<|im_end|>")
                elif msg["role"] == "assistant":
                    formatted_messages.append(f"<|im_start|>assistant\n{msg['content']}<|im_end|>")
            
            text = "\n".join(formatted_messages) + "\n<|im_start|>assistant\n"
            
        except Exception as e:
            print(f"Error formatting messages: {e}")
            return json.dumps({
                "answer": "Error",
                "idx": [],
                "explanation": f"Error formatting messages: {str(e)}"
            })

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_ids = model_inputs.input_ids

        eos_token_id_to_use = tokenizer.eos_token_id
        if isinstance(eos_token_id_to_use, list):
            eos_token_id_to_use = eos_token_id_to_use[0]
        if eos_token_id_to_use is None:
            eos_token_id_to_use = tokenizer.pad_token_id
            print("Warning: EOS token ID not found, using PAD token ID.")

        try:
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id_to_use,
                return_dict_in_generate=True,
                temperature=0.7,
                top_p=0.9
            )
        except Exception as e:
            print(f"Error during model generation: {e}")
            return json.dumps({
                "answer": "Error",
                "idx": [],
                "explanation": f"Error during generation: {str(e)}"
            })

        generated_sequence = output.sequences[0]
        full_generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        input_text_len = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
        generated_part = full_generated_text[input_text_len:].strip()

        # Remove any thinking process
        generated_part = re.sub(r'<think>.*?</think>', '', generated_part, flags=re.DOTALL)
        
        # Try to extract JSON content
        match = re.search(r'(\{[\s\S]*?\})', generated_part)
        if match:
            json_str = match.group(1).strip()
            try:
                # Enhanced JSON cleaning
                json_str = re.sub(r'^\{\s*,', '{', json_str)  # Remove leading comma
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Remove trailing commas
                json_str = re.sub(r'\n+', ' ', json_str)  # Replace newlines
                json_str = re.sub(r'\s+', ' ', json_str)  # Normalize whitespace
                json_str = re.sub(r'\\{2,}', r'\\', json_str)  # Fix excessive backslashes
                json_str = re.sub(r'"{2,}', '"', json_str)  # Fix multiple quotes
                json_str = re.sub(r'"\s*([,$$  $$])', r'"\1', json_str)  # Remove spaces before commas/brackets
                json_str = re.sub(r'([,$$  $$])\s*"', r'\1"', json_str)  # Remove spaces after commas/brackets
                json_str = re.sub(r'"\s*([a-zA-Z_]+)":', r'", "\1":', json_str)  # Add comma before keys
                json_str = re.sub(r'^{"\s*([a-zA-Z_]+)":', r'{"\1":', json_str)  # Fix start of JSON
                
                data = json.loads(json_str)
                if not isinstance(data, dict) or 'answer' not in data or 'idx' not in data or 'explanation' not in data:
                    print("Warning: Invalid JSON structure")
                    # Extract indices from explanation if possible
                    idx = re.findall(r'Premise (\d+)', data.get('explanation', ''))
                    idx = [int(i) for i in idx] if idx else []
                    return json.dumps({
                        "answer": data.get('answer', 'Uncertain'),
                        "idx": idx if data.get('answer', 'Uncertain') != 'Uncertain' else [],
                        "explanation": data.get('explanation', 'Invalid JSON structure')
                    })
                
                if data['answer'] not in ['Yes', 'No', 'Uncertain']:
                    data['answer'] = 'Uncertain'
                    data['explanation'] = f"Invalid answer value: {data['answer']}. " + data.get('explanation', '')
                
                if not isinstance(data['idx'], list):
                    idx = re.findall(r'Premise (\d+)', data.get('explanation', ''))
                    data['idx'] = [int(i) for i in idx] if idx else []
                    data['explanation'] = data.get('explanation', '') + " Invalid idx format, extracted from explanation."
                
                if data['answer'] == 'Uncertain' and data['idx']:
                    data['idx'] = []
                    data['explanation'] = data.get('explanation', '') + " idx reset to empty for Uncertain answer."
                
                return json.dumps(data)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Problematic JSON string: {json_str}")
                # Fallback: Extract fields manually
                answer_match = re.search(r'"answer":\s*"([^"]+)"', json_str)
                idx_match = re.search(r'"idx":\s*$$ ([^ $$]*)\]', json_str)
                explanation_match = re.search(r'"explanation":\s*"([^"]*)"', json_str)
                
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                    answer = answer_match.group(1).strip() if answer_match else 'Uncertain'
                    idx_str = idx_match.group(1).strip() if idx_match else ''
                    idx = [int(i.strip()) for i in idx_str.split(',') if i.strip().isdigit()] if idx_str else []
                    
                    if answer == 'Uncertain':
                        idx = []
                        explanation += " idx reset to empty for Uncertain answer."
                    elif not idx:
                        idx = [int(i) for i in re.findall(r'Premise (\d+)', explanation) if i.isdigit()]
                    
                    return json.dumps({
                        "answer": answer,
                        "idx": idx,
                        "explanation": explanation
                    })
                return json.dumps({
                    "answer": "Uncertain",
                    "idx": [],
                    "explanation": f"JSON decode error: {str(e)}. Raw output: {json_str}"
                })
        else:
            print(f"No valid JSON found in output: {generated_part}")
            # Fallback: Extract explanation and derive indices
            explanation = generated_part.strip()
            idx = [int(i) for i in re.findall(r'Premise (\d+)', explanation) if i.isdigit()]
            return json.dumps({
                "answer": "Uncertain",
                "idx": [],
                "explanation": f"Model output lacked JSON structure: {explanation}"
            })

def validate_answer_item(answer_item):
    """Validate that an answer item has all required fields with non-empty values"""
    required_fields = ['question', 'answer', 'idx', 'explanation']

    for field in required_fields:
        if field == 'idx':
             if field not in answer_item or not isinstance(answer_item[field], list):
                  print(f"Validation Error: Invalid or missing field: {field}")
                  return False
        elif field not in answer_item or not answer_item[field]:
            print(f"Validation Error: Missing or empty field: {field} in {answer_item}")
            return False

    if answer_item['idx'] and not all(isinstance(i, int) for i in answer_item['idx']):
         print(f"Validation Error: idx field contains non-integer elements: {answer_item['idx']}")
         return False

    if answer_item['answer'] == 'Uncertain' and answer_item['idx']:
         print(f"Validation Error: Answer is 'Uncertain' but idx list is not empty: {answer_item['idx']}")
         return False

    if 'timing' not in answer_item or not answer_item['timing']:
        print("Validation Warning: Missing timing information")
        return False

    return True

print(f"Starting script at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
model_load_start_time = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model(model_name = "Qwen/Qwen3-32B-AWQ"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model

model_name = "Qwen/Qwen3-32B-AWQ"
print(f"Loading model: {model_name}")
try:
    tokenizer, model = get_model(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    model.eval()

except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

model_load_time = time.time() - model_load_start_time
print(f"Model loading took {model_load_time:.2f} seconds")

train_yn_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_yn.json")
print(f"Loading dataset from: {train_yn_file}")

try:
    with open(train_yn_file, 'r', encoding='utf-8') as f:
        yn_ds = json.load(f)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

if not yn_ds:
    print("Dataset is empty. Exiting.")
    exit()

ds = yn_ds[42:43]
print(f"Loaded dataset with {len(yn_ds)} samples, processing first {len(ds)} samples (indices 0-49).")

answers = []
i = 0
total_start_time = time.time()
total_processing_time = 0

for sample_idx, sample in enumerate(ds):
    sample_start_time = time.time()
    print(f'\n{"="*80}')
    print(f'PROCESSING SAMPLE {sample_idx} (Original Index {i})')
    print(f'{"="*80}')

    premises_list = sample.get('premises-NL', [])
    if not isinstance(premises_list, list):
        print(f"Warning: 'premises-NL' in sample {sample_idx} is not a list. Using empty list.")
        premises_list = []

    question = sample.get('question', '')

    if not premises_list or not question:
         print(f"Warning: Missing premises or question for sample {sample_idx}. Skipping.")
         i+=1
         continue

    answer_item = {
        'question': question,
        'answer': '',
        'idx': [],
        'explanation': '',
        'timing': {},
        'original_index': i
    }
    final_output_str = None

    try:
        final_output_str, step_times = generate_pipeline(
            premises=premises_list,
            question=question,
            tokenizer=tokenizer,
            model=model,
            qs_type=QS_TYPE_YN,
            trace=False
        )
        answer_item['timing'] = step_times

        print(f"\nAttempting to directly parse final output for sample {sample_idx}...")
        response_json = None
        try:
            response_json = json.loads(final_output_str)
            print(f"Successfully parsed final output as JSON for sample {sample_idx}.")

            if not isinstance(response_json, dict) or \
               'answer' not in response_json or \
               'idx' not in response_json or \
               'explanation' not in response_json:
                 print(f"Warning [Sample {sample_idx}]: Parsed JSON is missing required keys. Treating as parse failure.")
                 response_json = None

        except json.JSONDecodeError as json_err:
            print(f"Direct JSON parsing failed for sample {sample_idx}: {json_err}")
            print("Raw output was:")
            print(final_output_str)
            response_json = None
        except Exception as load_err:
            print(f"An unexpected error occurred during json.loads for sample {sample_idx}: {load_err}")
            response_json = None

        if response_json:
            answer_item['answer'] = response_json.get('answer', 'Unknown')
            answer_item['idx'] = response_json.get('idx', [])
            answer_item['explanation'] = response_json.get('explanation', 'Explanation Missing Post-Parse')

            if not isinstance(answer_item['idx'], list):
                print(f"Warning [Sample {sample_idx}]: Parsed 'idx' is not a list ({type(answer_item['idx'])}). Resetting to [].")
                answer_item['idx'] = []
            else:
                try:
                    answer_item['idx'] = [int(x) for x in answer_item['idx']]
                except (ValueError, TypeError):
                    print(f"Warning [Sample {sample_idx}]: Could not convert all parsed idx items to int: {answer_item['idx']}. Resetting to [].")
                    answer_item['idx'] = []

            if not isinstance(answer_item['explanation'], str) or not answer_item['explanation']:
                answer_item['explanation'] = "Explanation Invalid Post-Parse"

            if answer_item['answer'] not in ['Yes', 'No', 'Uncertain']:
                print(f"Warning [Sample {sample_idx}]: Invalid answer value '{answer_item['answer']}'. Setting to Unknown.")
                answer_item['answer'] = 'Unknown'

            if answer_item['answer'] == 'Uncertain':
                if answer_item['idx']:
                    print(f"Warning [Sample {sample_idx}]: Correcting idx for Uncertain answer.")
                    answer_item['idx'] = []

            explanation = answer_item['explanation']

            import codecs
            explanation = codecs.decode(explanation, 'unicode_escape')
            explanation = explanation.replace('\n', ' ')
            explanation = explanation.replace('"', '"').replace('"', '"')  # fancy quotes to normal quotes
            explanation = explanation.replace("'", "'")  # fancy apostrophe to straight
            explanation = explanation.replace('→', '→')  # Already correct after decoding

            explanation = ' '.join(explanation.split())

            answer_item['explanation'] = explanation    

        else:
            #print(f"Setting error values for sample {sample_idx} due to JSON parse failure.")
            #answer_item['answer'] = 'Error: JSON Parse Failed'
            #answer_item['idx'] = []
            #answer_item['explanation'] = f"JSON Parse Failed. Raw Output: {final_output_str or '[No Raw Output]'}"
            
            fixed_str = final_output_str + '"}'
            cleaned_str = re.sub(r'^```json\n', '', fixed_str)
            data = json.loads(cleaned_str)

            answer_item['answer'] = repr(data['answer'])
            answer_item['idx'] = data['idx']
            explanation = data['explanation']

            import codecs
            explanation = codecs.decode(explanation, 'unicode_escape')

            explanation = explanation.replace('\n', ' ')
            explanation = explanation.replace('"', '"').replace('"', '"')  # fancy quotes to normal quotes
            explanation = explanation.replace("'", "'")  # fancy apostrophe to straight
            explanation = explanation.replace('→', '→')  # Already correct after decoding

            explanation = ' '.join(explanation.split())

            answer_item['explanation'] = explanation    
            answer_item.setdefault('timing', {'error': 'JSON parsing failed'})

    except Exception as e:
        import traceback
        print(f"Error during pipeline execution for sample {sample_idx} (Original Index {i}): {str(e)}")
        print(traceback.format_exc())
        answer_item['answer'] = 'Error: Pipeline Exception'
        answer_item['idx'] = []
        answer_item['explanation'] = f"Pipeline Error: {str(e)}\nRaw Output (if available): {final_output_str or '[N/A]'}"
        if 'timing' not in answer_item or not answer_item['timing']:
             answer_item['timing'] = {'error': f'Pipeline exception: {str(e)}'}

    if validate_answer_item(answer_item):
        print(f"Answer item {sample_idx} passed final validation.")
    else:
        print(f"WARNING: Answer item {sample_idx} (Original Index {i}) failed FINAL validation. Check data and logic.")

    answers.append(answer_item)
    i += 1

    sample_processing_time = time.time() - sample_start_time
    total_processing_time += sample_processing_time
    processed_count = sample_idx + 1
    avg_time_per_sample = total_processing_time / processed_count if processed_count > 0 else 0
    estimated_remaining_time = avg_time_per_sample * (len(ds) - processed_count)
    print(f"\nSample {sample_idx} processing time: {sample_processing_time:.2f} seconds")
    print(f"Average time per sample ({processed_count} processed): {avg_time_per_sample:.2f} seconds")
    print(f"Estimated remaining time for this run: {estimated_remaining_time:.2f} seconds ({estimated_remaining_time/60:.2f} minutes)")

print("\nSaving results...")
save_start_time = time.time()
output_dir = "/home/manh/Projects/QAChallenge/"
output_file = os.path.join(output_dir, "yn_answer_kanz_test_50_direct_json.json")

try:
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_file}")
except Exception as e:
    print(f"Error saving results to {output_file}: {e}")

save_time = time.time() - save_start_time
total_time = time.time() - total_start_time

print(f"Final save took {save_time:.2f} seconds")
print(f"Total execution time for {len(ds)} samples: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Model loading time: {model_load_time:.2f} seconds")
print(f"Data processing time: {total_processing_time:.2f} seconds")
avg_time = total_processing_time / len(ds) if ds else 0
print(f"Average time per sample: {avg_time:.2f} seconds")
print(f"Script completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
