import torch
import json
import os
from typing import List, Dict, Any
import re
import time
from datetime import datetime
import xml.etree.ElementTree as ET

'''
PROMPT_COT_SYS_YN ="""
You are a helpful assistant. Given premises and a Yes/No/Uncertain question,
reason step-by-step using only the provided premises to determine the answer.
Focus on logical deduction and clearly state which premises support each step.
If the premises do not contain sufficient information to answer the question with certainty, 
your answer should be 'Uncertain' with an empty list of premise indices.
"""
'''

PROMPT_COT_SYS_YN ="""
You are a helpful assistant. Given premises and a Yes/No/Uncertain question,
reason step-by-step using ONLY the minimal set of premises necessary to answer the question.
Focus on logical deduction and clearly state which premises support each step.
Avoid including premises that are not directly required to derive the answer.
If the premises do not contain sufficient information to answer the question with certainty, 
your answer should be 'Uncertain' with an empty list of premise indices.
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
Now, based on the following information:
- Concise Answer: '<?concise_answer>'
- Identified Premise Indices: <?idx_list_str> # String representation, e.g., "1,3,5" or ""
- Step-by-Step Explanation: "<?explanation>"

Construct the final JSON object EXACTLY in the format below. Output ONLY the JSON object and nothing else.

Response in the following XML format:
```
<response>
    <answer>{answer}</answer>
    <idx>{idx}</idx>
    <explanation>{explanation}</explanation>
</response>
```
Field description:
    - `{answer}` is concise answer, restricted in 'Yes' , 'No' , 'Uncertain'
    - `{idx}` list the numerical indexes of the premises (from `Premise #X`) that support the chosen answer.
    - `{explanation}` is your reasoning within <think> element expressed in text with only text, clearly refer relevant premises.
```
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

    if final_output_str:
        parsed_output = parse_xml_output(final_output_str)
        if parsed_output:
            return parsed_output, step_times
        else:
            return None, step_times
    else:
        return None, step_times

def generate_answer(tokenizer, model, messages, max_tokens=1000):
    with torch.no_grad():
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
             print("Error: 'messages' must be a list of dictionaries.")
             return "Error in message format"

        valid_messages = [m for m in messages if m.get("content")]
        if not valid_messages:
            print("Error: No valid messages to process.")
            return "Error: No valid messages"

        try:
            text = tokenizer.apply_chat_template(
                valid_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Error applying chat template: {e}")
            print(f"Problematic messages: {valid_messages}")
            return f"Error applying template: {e}"

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
             return f"Error during generation: {e}"

    generated_sequence = output.sequences[0]
    full_generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)

    input_text_len = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    generated_part = full_generated_text[input_text_len:].strip()

    # Look for XML response
    match = re.search(r"<response>([\s\S]*?)</response>", generated_part)
    if match:
        print("Found XML response format")
        return match.group(0).strip()
    else:
        return generated_part

def parse_xml_output(output_str: str) -> Dict:
    """Parse XML output from the model"""
    try:
        # Clean up the XML string
        xml_str = output_str.strip()
        if not xml_str.startswith('<response>'):
            xml_str = f'<response>{xml_str}</response>'
        
        # Parse XML
        root = ET.fromstring(xml_str)
        
        # Extract values
        answer = root.find('answer').text if root.find('answer') is not None else 'Unknown'
        idx_text = root.find('idx').text if root.find('idx') is not None else ''
        explanation = root.find('explanation').text if root.find('explanation') is not None else ''
        
        # Convert idx string to list of integers
        idx_list = []
        if idx_text:
            try:
                idx_list = [int(x.strip()) for x in idx_text.split(',') if x.strip().isdigit()]
            except ValueError:
                print(f"Warning: Could not parse idx values: {idx_text}")
        
        return {
            "answer": answer,
            "idx": idx_list,
            "explanation": explanation
        }
    except Exception as e:
        print(f"Error parsing XML output: {e}")
        return None

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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading model: {model_name}")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        quantization_config = BitsAndBytesConfig(load_in_8bit=True,llm_int8_threshold=6.0),
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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

        print(f"\nAttempting to parse final output for sample {sample_idx}...")
        if final_output_str:
            answer_item['answer'] = final_output_str.get('answer', 'Unknown')
            answer_item['idx'] = final_output_str.get('idx', [])
            answer_item['explanation'] = final_output_str.get('explanation', 'Explanation Missing Post-Parse')

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

            # Clean up explanation text
            explanation = answer_item['explanation']
            import codecs
            explanation = codecs.decode(explanation, 'unicode_escape')
            explanation = explanation.replace('\n', ' ')
            explanation = explanation.replace('"', '"').replace('"', '"')
            explanation = explanation.replace("'", "'")
            explanation = explanation.replace('→', '→')
            explanation = ' '.join(explanation.split())
            answer_item['explanation'] = explanation
        else:
            print(f"Setting error values for sample {sample_idx} due to parse failure.")
            answer_item['answer'] = 'Error: Parse Failed'
            answer_item['idx'] = []
            answer_item['explanation'] = f"Parse Failed. Raw Output: {final_output_str or '[No Raw Output]'}"
            answer_item.setdefault('timing', {'error': 'Parsing failed'})

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
output_file = os.path.join(output_dir, "yn_answer_mq_test_50_direct_json.json")

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
