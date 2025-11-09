import torch
import json
import os
from typing import Any
import re
import time
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
Review the step-by-step explanation:
"<?explanation>"

List the numeric indices of all premises explicitly mentioned or cited in the explanation.
Output *only* the numbers, separated by commas (e.g., 1,3,5).
If no premises were cited or the answer is 'Uncertain' because the premises are insufficient, leave this empty.
"""

PROMPT_COT_USER_FINAL_YN = """
Now, based on the following information:
- Concise Answer: '<?concise_answer>'
- Premise Indices: <?idx_list>
- Step-by-Step Explanation: "<?explanation>"

Assemble the final answer in the specified JSON format:

```json
{
  "answer": "<?concise_answer>",
  "idx": [<?idx_list>],
  "explanation": "<?explanation_escaped>"
}
```
"""
QS_TYPE_YN = 'Yes/No/Uncertain'

def generate_prompt(template, dict):
    prompt = template

    for key, val in dict.items():
        prompt = prompt.replace(f'<?{key}>', val)

    return prompt

def create_sys_prompt(qs_type):
    """
    Create input for system to answer the question
    """
    if qs_type == QS_TYPE_YN:
        return generate_prompt(PROMPT_COT_SYS_YN, dict={})
    else:
        print(f"Warning: Unexpected question type '{qs_type}' in create_sys_prompt")
        return "" 

def create_user_prompt(premises, question):
    """
    create input for user to answer the question
    """
    dict = {}
    premise_vals = ""
    for id, premise in premises.items():
        premise_vals += f"    - Premise {id}: {premise}\n"
    dict['premises'] = premise_vals
    dict['question'] = question
    return generate_prompt(PROMPT_COT_USER_STEP_1, dict=dict)

def create_user_step_2(qs_type):
    # Simplified as we only handle YN type now
    if qs_type == QS_TYPE_YN:
        return generate_prompt(PROMPT_COT_USER_STEP_2_YN, {})
    else:
        print(f"Warning: Unexpected question type '{qs_type}' in create_user_step_2")
        return ""

def create_user_step_3_explanation(concise_answer):
    """
    Create prompt for step 3 explanation with the concise answer.
    """
    return generate_prompt(PROMPT_COT_USER_STEP_3_EXPLANATION, {'concise_answer': concise_answer})

def create_user_step_4_idx(explanation):
    """
    Create prompt for step 4 to extract premise indices from the explanation.
    """
    return generate_prompt(PROMPT_COT_USER_STEP_4_IDX, {'explanation': explanation})

def create_user_final(concise_answer, idx_list, explanation, qs_type=QS_TYPE_YN):
    """
    Create prompt for final step with all required components.
    """
    # Escape explanation for JSON
    explanation_escaped = explanation.replace('"', '\\"').replace('\n', '\\n')
    
    if qs_type == QS_TYPE_YN:
        return generate_prompt(PROMPT_COT_USER_FINAL_YN, {
            'concise_answer': concise_answer,
            'idx_list': idx_list,
            'explanation': explanation,
            'explanation_escaped': explanation_escaped
        })
    else:
        print(f"Warning: Unexpected question type '{qs_type}' in create_user_final")
        return ""

def generate_pipeline(premises, question, tokenizer, model, qs_type = QS_TYPE_YN, trace=False): # Default to YN
    pipeline_start_time = time.time()
    step_times = {}

    # Question type detection is removed as we focus on QS_TYPE_YN
    print(f"Using fixed question type: {qs_type}")
    if qs_type != QS_TYPE_YN:
        print(f"Error: This pipeline is configured only for '{QS_TYPE_YN}' questions.")
        return None, [], {'error': f"Unsupported question type: {qs_type}"}

    # Prepare initial messages
    sys = create_sys_prompt(qs_type)
    user = create_user_prompt(premises=premises, question=question)
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user}
    ]

    # Step 1: Generate initial reasoning
    if trace:
        print('Step 1: Generating initial reasoning...')
    
    step_1_start_time = time.time()
    step_1_reasoning = generate_answer(tokenizer=tokenizer, model=model, messages=messages, max_tokens=800)
    step_times['step_1'] = time.time() - step_1_start_time
    
    if trace:
        print(f"Step 1 took {step_times['step_1']:.2f} seconds")

    # Step 2: Extract concise answer
    if trace:
        print('Step 2: Extracting concise answer...')
    
    step_2_start_time = time.time()
    messages.append({"role": "system", "content": step_1_reasoning})
    messages.append({"role": "user", "content": create_user_step_2(qs_type)})
    
    step_2_answer_raw = generate_answer(tokenizer=tokenizer, model=model, messages=messages, max_tokens=200)
    # Extract just "Yes", "No", or "Uncertain" from response
    match = re.search(r'\b(Yes|No|Uncertain)\b', step_2_answer_raw, re.IGNORECASE)
    concise_answer = match.group(1).capitalize() if match else "Uncertain"
    step_times['step_2'] = time.time() - step_2_start_time
    
    if trace:
        print(f"Step 2 took {step_times['step_2']:.2f} seconds")
        print(f"Concise answer: {concise_answer}")

    # Step 3: Generate step-by-step explanation based on concise answer
    if trace:
        print('Step 3: Generating step-by-step explanation...')
    
    step_3_start_time = time.time()
    messages.append({"role": "system", "content": step_2_answer_raw})
    messages.append({"role": "user", "content": create_user_step_3_explanation(concise_answer)})
    
    step_3_explanation = generate_answer(tokenizer=tokenizer, model=model, messages=messages, max_tokens=500)
    step_times['step_3'] = time.time() - step_3_start_time

    if trace:
        print(f"Step 3 took {step_times['step_3']:.2f} seconds")

    # Step 4: Extract premise indices from explanation
    if trace:
        print('Step 4: Extracting premise indices from explanation...')
    
    step_4_start_time = time.time()
    messages.append({"role": "system", "content": step_3_explanation})
    messages.append({"role": "user", "content": create_user_step_4_idx(step_3_explanation)})
    
    step_4_indices = generate_answer(tokenizer=tokenizer, model=model, messages=messages, max_tokens=200)
    step_times['step_4'] = time.time() - step_4_start_time

    if trace:
        print(f"Step 4 took {step_times['step_4']:.2f} seconds")
        print(f"Extracted indices: {step_4_indices}")

    # Step 5: Final structured answer with all components
    if trace:
        print('Step 5: Generating final structured answer...')
    
    final_start_time = time.time()
    messages.append({"role": "system", "content": step_4_indices})
    messages.append({"role": "user", "content": create_user_final(concise_answer, step_4_indices, step_3_explanation, qs_type)})
    
    final_answer = generate_answer(tokenizer=tokenizer, model=model, messages=messages, max_tokens=800)
    step_times['final_step'] = time.time() - final_start_time

    if trace:
        print(f"Final step took {step_times['final_step']:.2f} seconds")
    
    total_time = time.time() - pipeline_start_time
    step_times['total_pipeline'] = total_time
    
    if trace:
        print(f"Total pipeline execution time: {total_time:.2f} seconds")
        print("Step-by-step timing:")
        for step, duration in step_times.items():
            print(f"  {step}: {duration:.2f} seconds")

    return final_answer, messages, step_times
    
def generate_answer(tokenizer, model, messages, max_tokens=1000):
    with torch.no_grad():
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_ids = model_inputs.input_ids

        # Optimize generation parameters
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,  # Use the specified token limit
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            # Add temperature and top_p parameters for more focused generation
            temperature=0.7,
            top_p=0.9
        )

    generated_sequence = output.sequences[0]
    full_generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    
    input_text_len = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    return full_generated_text[input_text_len:].strip()

def parse_res(res_json):
    print("\nParsing response JSON...")
    res_json = res_json.replace('\n', '')
    matchs = re.findall(r"```json\s*([\s\S]*?)\s*```", res_json)
    if matchs:
        try:
            parsed = json.loads(matchs[0])
            print("Successfully parsed JSON response")
            return parsed
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print("Raw JSON content:")
            print(matchs[0])
            
            # If JSON parsing fails, try to extract the answer directly
            answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', res_json)
            idx_match = re.findall(r'"idx"\s*:\s*\[([^\]]*)\]', res_json)
            explanation_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', res_json)
            
            if answer_match and idx_match and explanation_match:
                answer = answer_match.group(1)
                idx_str = idx_match[0]
                idx = [int(i.strip()) for i in idx_str.split(',') if i.strip().isdigit()]
                explanation = explanation_match.group(1)
                
                print("Extracted answer components using regex")
                return {
                    "answer": answer,
                    "idx": idx,
                    "explanation": explanation
                }
    else:
        print("No JSON found in response")
        
        # Try to extract structured data from the raw response
        answer_match = re.search(r'answer["\s:]+([^"\n]+)', res_json, re.IGNORECASE)
        idx_match = re.findall(r'idx["\s:]+\[([^\]]+)\]', res_json, re.IGNORECASE)
        explanation_match = re.search(r'explanation["\s:]+([^"\n]+)', res_json, re.IGNORECASE)
        
        if answer_match and idx_match and explanation_match:
            answer = answer_match.group(1).strip()
            idx_str = idx_match[0]
            idx = [int(i.strip()) for i in idx_str.split(',') if i.strip().isdigit()]
            explanation = explanation_match.group(1).strip()
            
            print("Extracted answer components from raw response")
            return {
                "answer": answer,
                "idx": idx,
                "explanation": explanation
            }
    
    return None

def validate_answer_item(answer_item):
    """Validate that an answer item has all required fields with non-empty values"""
    required_fields = ['question', 'answer', 'idx', 'explanation']
    
    # Check if all required fields exist and are not empty
    for field in required_fields:
        if field not in answer_item or not answer_item[field]:
            print(f"Missing or empty field: {field}")
            return False
    
    # Check if idx is a list and not empty
    if not isinstance(answer_item['idx'], list):
        print("idx field is not a list")
        return False
    
    # Empty idx is allowed if answer is Uncertain
    if answer_item['answer'] != 'Uncertain' and not answer_item['idx']:
        print("idx field is empty for a non-Uncertain answer")
        return False
    
    # Check if timing information exists
    if 'timing' not in answer_item or not answer_item['timing']:
        print("Missing timing information")
        return False
    
    return True

# Load model and tokenizer
print(f"Starting script at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
model_load_start_time = time.time()

model_name = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    quantization_config = BitsAndBytesConfig(load_in_8bit=True,llm_int8_threshold=6.0),
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model_load_time = time.time() - model_load_start_time
print(f"Model loading took {model_load_time:.2f} seconds")

train_yn_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_yn.json")
print(f"Loading dataset from: {train_yn_file}")

with open(train_yn_file, 'r', encoding='utf-8') as f:
    yn_ds = json.load(f)

ds = yn_ds[43:44]
print(f"Loaded dataset with {len(yn_ds)} samples, but will process only {len(ds)} samples")

answers = []
i = 0
total_start_time = time.time()
total_processing_time = 0

for sample in ds:
    sample_start_time = time.time()
    print(f'\n{"="*80}')
    print(f'PROCESSING SAMPLE {i}')
    print(f'{"="*80}')
    i+=1
    premises = sample.get('premises-NL', {})
    question = sample.get('question', '')

    # Initialize answer item with all required fields
    answer_item = { 
        'question': question, 
        'answer': '', 
        'idx': [], 
        'explanation': '', 
        'timing': {},
    }
    
    try:
        premise_dict = {str(i+1): premises[i] for i in range(len(premises))}
        res, messages, step_times = generate_pipeline(premises=premise_dict, question=question, tokenizer=tokenizer, model=model, qs_type=QS_TYPE_YN, trace=True)
        answer_item['timing'] = step_times
        
        print("\nParsing final response...")
        response_json = parse_res(res)

        if response_json:
            # Ensure all required fields are present
            answer_item['answer'] = response_json.get('answer', '')
            answer_item['idx'] = list(set(response_json.get('idx', [])))
            answer_item['explanation'] = response_json.get('explanation', '')
            
            # Validate the answer item
            if validate_answer_item(answer_item):
                print(f"Final answer: {answer_item['answer']}")
                print(f"Used premises: {answer_item['idx']}")
                print("Answer item validated successfully")
            else:
                print("Answer item validation failed, attempting to fix...")
                
                # If answer is missing but we have a response, try to extract it
                if not answer_item['answer'] and res:
                    # Try to extract a simple Yes/No answer
                    yes_no_match = re.search(r'\b(yes|no|uncertain)\b', res.lower())
                    if yes_no_match:
                        answer_item['answer'] = yes_no_match.group(1).capitalize()
                        print(f"Extracted answer: {answer_item['answer']}")
                
                # If idx is empty, try to extract premise numbers
                if not answer_item['idx'] and res:
                    idx_matches = re.findall(r'premise\s+(\d+)', res.lower())
                    if idx_matches:
                        answer_item['idx'] = list(set([int(idx) for idx in idx_matches]))
                        print(f"Extracted premise indices: {answer_item['idx']}")
                
                # If explanation is empty, use the raw response
                if not answer_item['explanation'] and res:
                    answer_item['explanation'] = res
                    print("Using raw response as explanation")
        else:
            print("Failed to parse response JSON")
            
            # Try to extract basic information from the raw response
            if res:
                # Try to extract a simple Yes/No answer
                yes_no_match = re.search(r'\b(yes|no|uncertain)\b', res.lower())
                if yes_no_match:
                    answer_item['answer'] = yes_no_match.group(1).capitalize()
                    print(f"Extracted answer from raw response: {answer_item['answer']}")
                
                # Try to extract premise numbers
                idx_matches = re.findall(r'premise\s+(\d+)', res.lower())
                if idx_matches:
                    answer_item['idx'] = list(set([int(idx) for idx in idx_matches]))
                    print(f"Extracted premise indices from raw response: {answer_item['idx']}")
                
                # Use the raw response as explanation
                answer_item['explanation'] = res
                print("Using raw response as explanation")
    except Exception as e:
        print(f"Error processing sample: {str(e)}")
        
        # Try to set default values for required fields
        if not answer_item['answer']:
            answer_item['answer'] = 'Unknown'
        if not answer_item['idx']:
            answer_item['idx'] = []
        if not answer_item['explanation']:
            answer_item['explanation'] = f"Error occurred: {str(e)}"
        if not answer_item['timing']:
            answer_item['timing'] = {'error': 'No timing information available'}

    # Final validation before adding to answers
    if validate_answer_item(answer_item):
        print("Answer item is complete and valid")
    else:
        print("WARNING: Answer item is incomplete or invalid")
        # Add default values for any missing fields
        if not answer_item['answer']:
            answer_item['answer'] = 'Unknown'
        if not answer_item['idx']:
            answer_item['idx'] = []
        if not answer_item['explanation']:
            answer_item['explanation'] = 'No explanation available'
        if not answer_item['timing']:
            answer_item['timing'] = {'error': 'No timing information available'}

    answers.append(answer_item)
    
    sample_processing_time = time.time() - sample_start_time
    total_processing_time += sample_processing_time
    avg_time_per_sample = total_processing_time / i
    estimated_remaining_time = avg_time_per_sample * (len(ds) - i)
    
    print(f"\nSample {i-1} processing time: {sample_processing_time:.2f} seconds")
    print(f"Average time per sample: {avg_time_per_sample:.2f} seconds")
    print(f"Estimated remaining time: {estimated_remaining_time:.2f} seconds ({estimated_remaining_time/60:.2f} minutes)")

# Save results to yn_answer_mq.json
print("\nSaving results...")
save_start_time = time.time()
output = "/home/manh/Projects/QAChallenge/yn_answer_mq_test_single_optimized.json"

# Validate all answer items before saving
valid_answers = []
for idx, answer_item in enumerate(answers):
    if validate_answer_item(answer_item):
        valid_answers.append(answer_item)
    else:
        print(f"WARNING: Answer item {idx} is incomplete or invalid, attempting to fix...")
        # Add default values for any missing fields
        if not answer_item['answer']:
            answer_item['answer'] = 'Unknown'
        if not answer_item['idx']:
            answer_item['idx'] = []
        if not answer_item['explanation']:
            answer_item['explanation'] = 'No explanation available'
        if not answer_item['timing']:
            answer_item['timing'] = {'error': 'No timing information available'}
        valid_answers.append(answer_item)

# Save the validated answers
with open(output, "w", encoding="utf-8") as f:
    json.dump(valid_answers, f, ensure_ascii=False, indent=2)

save_time = time.time() - save_start_time
total_time = time.time() - total_start_time

print(f"Results saved to: {output}")
print(f"Final save took {save_time:.2f} seconds")
print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Model loading time: {model_load_time:.2f} seconds")
print(f"Data processing time: {total_processing_time:.2f} seconds")
print(f"Average time per sample: {total_processing_time/len(ds):.2f} seconds")
print(f"Script completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
