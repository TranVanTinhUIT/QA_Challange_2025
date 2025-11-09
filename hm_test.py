from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
import os
import time
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-Math-7B"
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0),
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
except Exception as e:
    logger.error(f"Failed to load model or tokenizer: {e}")
    exit(1)

# Load dataset
train_hm_file = "/home/manh/Projects/QAChallenge/train_hm.json"
try:
    with open(train_hm_file, 'r', encoding='utf-8') as f:
        hm_ds = json.load(f)
except FileNotFoundError:
    logger.error(f"Dataset file not found: {train_hm_file}")
    exit(1)
except json.JSONDecodeError:
    logger.error(f"Invalid JSON in dataset file: {train_hm_file}")
    exit(1)

# Use full dataset or subset as needed
ds = hm_ds[0:1]
logger.info(f"Loaded dataset with {len(hm_ds)} samples")
logger.info(f"Processing {len(ds)} samples.")

# --- Multi-Step Prompts ---

# Step 1: Problem formulation
PROMPT_STEP1_SYS = """
You are a math teacher who creates clear, concise problems based on provided premises.
Take the premises and question, and reformulate them into a coherent, standalone mathematical problem.
"""

PROMPT_STEP1_USER = """
Here are the premises:
<?premises>

Question: <?question>

Create a clear, standalone math problem that incorporates these premises and leads to answering this question.
"""

# Step 2: Problem solving
PROMPT_STEP2_SYS = """
You are a precise logical reasoning assistant specialized in solving 'How Many' math problems.
Solve the given problem step-by-step, showing your reasoning process.
Be thorough and precise in your calculations.
"""

PROMPT_STEP2_USER = """
Problem: <?problem>

Solve this problem step-by-step. Make sure to:
1. Identify the key information
2. Show your reasoning process
3. Calculate the final answer
4. Clearly state which premises you used in your solution

Be detailed and explicit in your explanation.
"""

# Step 3: Extract premise indices
PROMPT_STEP3_SYS = """
You are a precise analysis assistant. Your task is to identify which premise numbers were used in solving a math problem.
"""

PROMPT_STEP3_USER = """
Original Premises:
<?premises>

Solution explanation: <?explanation>

List only the premise numbers (as integers) that were used in the solution explanation.
Format your response as a comma-separated list of integers.
"""

# Step 4: Final synthesis
PROMPT_STEP4_SYS = """
You are a precise summarization assistant. Create a concise explanation based on the detailed solution.
"""

PROMPT_STEP4_USER = """
Detailed explanation: <?detailed_explanation>
Answer: <?answer>
Premises used: <?premises_used>

Create a concise explanation (50 words or fewer) that explains how the answer was derived using the premises.
"""

# Original combined prompt kept for reference
PROMPT_HM_COMBINED_SYS = """
You are a precise logical reasoning assistant specialized in answering 'How Many' questions based *only* on provided premises.
Follow these steps:
1. Analyze the premises to identify all relevant information (entities, quantities, relationships) needed to answer the question.
2. Perform step-by-step reasoning and calculations based *strictly* on the premises. Clearly show your work.
3. Cite the premise number(s) used for each step (e.g., "Using Premise 1 and 3...").
4. Determine the final numerical answer.
5. If the answer cannot be determined solely from the premises, the answer must be "Uncertain".
6. Construct a JSON object containing the final answer, the list of premise indices used in your reasoning, and the detailed step-by-step explanation.

Output ONLY the JSON object in the exact format below:
```json
{
  "answer": "<The final numerical answer as a string, or 'Uncertain'>",
  "idx": [<List of integer indices of premises used in the explanation>],
  "explanation": "<A string containing the detailed step-by-step reasoning, including premise citations and calculations. Ensure proper JSON string escaping.>"
}
```
"""

PROMPT_HM_COMBINED_USER = """
Premises:
<?premises>

Question: <?question>

Return the answer in this JSON format:
```json
{
"answer": "<your answer by words or numbers>",
"idx": <relevant premise numbers as integers>,
"explanation": "<your explanation in 50 words or fewer how you got the answer using the premises.>"
}
```
"""

# --- Utility Functions ---
def generate_prompt(template, dict):
    """Generate a prompt by replacing placeholders with values."""
    prompt = template
    for key, val in dict.items():
        prompt = prompt.replace(f'<?{key}>', val)
    return prompt

def generate_answer(tokenizer, model, messages, max_tokens=1000):
    """Generate answer from the model with optimized parameters."""
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
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            temperature=0.7,
            top_p=0.9
        )

        generated_sequence = output.sequences[0]
        full_generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        input_text_len = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
        return full_generated_text[input_text_len:].strip()

def generate_pipeline(premises, question, tokenizer, model, trace=True):
    """Generate answer using a single-shot pipeline. Kept as fallback."""
    pipeline_start_time = time.time()
    step_times = {}

    # Prepare premises string
    premise_vals = ""
    for id, premise in premises.items():
        premise_vals += f"    - Premise {id}: {premise}\n"

    # Create prompts
    sys_prompt = generate_prompt(PROMPT_HM_COMBINED_SYS, {})
    user_prompt = generate_prompt(PROMPT_HM_COMBINED_USER, {
        'premises': premise_vals,
        'question': question
    })

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Single Generation Step
    if trace:
        logger.info('Generating final JSON answer...')
    generation_start_time = time.time()
    # Increased max_tokens for combined reasoning and JSON structure
    final_json_str = generate_answer(tokenizer=tokenizer, model=model, messages=messages, max_tokens=1200)
    generation_time = time.time() - generation_start_time
    step_times['json_generation'] = generation_time
    if trace:
        logger.info(f"JSON Generation took {generation_time:.2f} seconds")

    total_time = time.time() - pipeline_start_time
    step_times['total_pipeline'] = total_time
    if trace:
        logger.info(f"Total pipeline execution time: {total_time:.2f} seconds")

    return final_json_str, step_times

def generate_multi_step_pipeline(premises, question, tokenizer, model, trace=True):
    """Generate answer using a multi-step pipeline as requested."""
    pipeline_start_time = time.time()
    step_times = {}

    # Prepare premises string
    premise_vals = ""
    for id, premise in premises.items():
        premise_vals += f"    - Premise {id}: {premise}\n"

    # Step 1: Problem formulation
    if trace:
        logger.info('Step 1: Formulating problem from premises and question...')
    step1_start_time = time.time()
    
    step1_messages = [
        {"role": "system", "content": PROMPT_STEP1_SYS},
        {"role": "user", "content": generate_prompt(PROMPT_STEP1_USER, {
            'premises': premise_vals,
            'question': question
        })}
    ]
    
    problem_formulation = generate_answer(tokenizer=tokenizer, model=model, messages=step1_messages, max_tokens=800)
    step_times['step1_problem_formulation'] = time.time() - step1_start_time
    
    if trace:
        logger.info(f"Step 1 completed in {step_times['step1_problem_formulation']:.2f} seconds")
        logger.info(f"Problem formulation: {problem_formulation}")

    # Step 2: Problem solving
    if trace:
        logger.info('Step 2: Solving the problem with detailed reasoning...')
    step2_start_time = time.time()
    
    step2_messages = [
        {"role": "system", "content": PROMPT_STEP2_SYS},
        {"role": "user", "content": generate_prompt(PROMPT_STEP2_USER, {
            'problem': problem_formulation
        })}
    ]
    
    detailed_solution = generate_answer(tokenizer=tokenizer, model=model, messages=step2_messages, max_tokens=1200)
    step_times['step2_problem_solving'] = time.time() - step2_start_time
    
    if trace:
        logger.info(f"Step 2 completed in {step_times['step2_problem_solving']:.2f} seconds")
        logger.info(f"Detailed solution: {detailed_solution}")
    
    # Extract answer from detailed solution
    answer_pattern = r"(?:the answer is|final answer is|answer:|result is)[:\s]+([^\.\n]+)"
    answer_matches = re.findall(answer_pattern, detailed_solution, re.IGNORECASE)
    answer = answer_matches[0].strip() if answer_matches else "Uncertain"
    
    if trace:
        logger.info(f"Extracted answer: {answer}")

    # Step 3: Extract premise indices
    if trace:
        logger.info('Step 3: Extracting premise indices from solution...')
    step3_start_time = time.time()
    
    step3_messages = [
        {"role": "system", "content": PROMPT_STEP3_SYS},
        {"role": "user", "content": generate_prompt(PROMPT_STEP3_USER, {
            'premises': premise_vals,
            'explanation': detailed_solution
        })}
    ]
    
    premise_indices_raw = generate_answer(tokenizer=tokenizer, model=model, messages=step3_messages, max_tokens=300)
    step_times['step3_extract_indices'] = time.time() - step3_start_time
    
    # Parse and validate premise indices
    idx_pattern = r'(\d+)'
    idx_matches = re.findall(idx_pattern, premise_indices_raw)
    premise_indices = []
    for idx in idx_matches:
        try:
            premise_idx = int(idx)
            if premise_idx > 0 and premise_idx <= len(premises):
                premise_indices.append(premise_idx)
        except ValueError:
            continue
    
    premise_indices = sorted(list(set(premise_indices)))
    
    if trace:
        logger.info(f"Step 3 completed in {step_times['step3_extract_indices']:.2f} seconds")
        logger.info(f"Extracted premise indices: {premise_indices}")

    # Step 4: Final concise explanation
    if trace:
        logger.info('Step 4: Creating concise explanation...')
    step4_start_time = time.time()
    
    step4_messages = [
        {"role": "system", "content": PROMPT_STEP4_SYS},
        {"role": "user", "content": generate_prompt(PROMPT_STEP4_USER, {
            'detailed_explanation': detailed_solution,
            'answer': answer,
            'premises_used': ", ".join([str(idx) for idx in premise_indices])
        })}
    ]
    
    concise_explanation = generate_answer(tokenizer=tokenizer, model=model, messages=step4_messages, max_tokens=300)
    step_times['step4_concise_explanation'] = time.time() - step4_start_time
    
    if trace:
        logger.info(f"Step 4 completed in {step_times['step4_concise_explanation']:.2f} seconds")
        logger.info(f"Concise explanation: {concise_explanation}")

    # Construct final JSON
    result = {
        "answer": answer,
        "idx": premise_indices,
        "explanation": concise_explanation.strip()
    }
    
    total_time = time.time() - pipeline_start_time
    step_times['total_pipeline'] = total_time
    
    if trace:
        logger.info(f"Total multi-step pipeline execution time: {total_time:.2f} seconds")
        logger.info(f"Final result: {result}")

    # Convert to JSON string (will be parsed later by existing code)
    final_json = json.dumps(result, ensure_ascii=False)
    return f"```json\n{final_json}\n```", step_times

def parse_final_json(json_str):
    """Extract JSON object from the model's final output string."""
    logger.info("Attempting to parse final JSON output...")
    matches = re.findall(r"```json\s*([\s\S]*?)\s*```", json_str)
    content_to_parse = matches[0] if matches else json_str
    
    try:
        # Clean potential leading/trailing whitespace or newlines
        cleaned_content = content_to_parse.strip()
        parsed = json.loads(cleaned_content)
        logger.info("Successfully parsed JSON response")
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError parsing final output: {e}")
        logger.error(f"Raw content received: {json_str}")
    except Exception as e:
        logger.error(f"Unexpected error parsing final output: {e}")
        logger.error(f"Raw content received: {json_str}")
    return None

def validate_answer_item(answer_item):
    """Validate that an answer item has all required fields with non-empty values."""
    required_fields = ['question', 'answer', 'idx', 'explanation', 'timing']
    for field in required_fields:
        if field not in answer_item or not answer_item[field]:
            logger.warning(f"Missing or empty field: {field}")
            return False
    # Allow empty idx list if answer is Uncertain or no premises were needed
    if not isinstance(answer_item['idx'], list):
         logger.warning("idx field is not a list")
         return False
    if answer_item['answer'] != 'Uncertain' and not answer_item['idx']:
        # Only warn if answer is not uncertain and idx is empty
        logger.warning("idx field is empty for a non-Uncertain answer.")
    return True

# Process dataset
answers = []
total_start_time = time.time()
total_processing_time = 0
i = 0

for sample in ds:
    sample_start_time = time.time()
    logger.info(f'\n{"="*80}')
    logger.info(f'PROCESSING SAMPLE {i}')
    logger.info(f'{"="*80}')
    i += 1
    premises = sample.get('premises-NL', [])
    question = sample.get('question', '')

    # Validate sample
    if not premises or not question:
        logger.warning(f"Invalid sample {i}: Missing premises or question")
        answers.append({
            'question': question,
            'answer': '',
            'idx': [],
            'explanation': 'Invalid sample',
            'timing': {'error': 'Invalid sample'}
        })
        continue

    answer_item = {
        'question': question,
        'answer': '',
        'idx': [],
        'explanation': '',
        'timing': {}
    }

    try:
        premise_dict = {str(i+1): premise for i, premise in enumerate(premises)}
        # Use the new multi-step pipeline instead of the original one
        final_json_str, step_times = generate_multi_step_pipeline(
            premises=premise_dict,
            question=question,
            tokenizer=tokenizer,
            model=model,
            trace=True
        )
        answer_item['timing'] = step_times

        # Parse the direct JSON output
        response_json = parse_final_json(final_json_str)

        if response_json:
            # Ensure 'idx' contains only integers and is sorted
            raw_idx = response_json.get('idx', [])
            processed_idx = []
            if isinstance(raw_idx, list):
                for item in raw_idx:
                    try:
                        processed_idx.append(int(item))
                    except (ValueError, TypeError):
                        logger.warning(f"Non-integer value found in idx list: {item}. Skipping.")
            else:
                 logger.warning(f"Parsed 'idx' is not a list ({type(raw_idx)}). Setting to [].")

            answer_item['answer'] = str(response_json.get('answer', 'Error: Missing Answer'))
            answer_item['idx'] = sorted(list(set(processed_idx)))
            answer_item['explanation'] = response_json.get('explanation', 'Error: Missing Explanation')

            if validate_answer_item(answer_item):
                logger.info(f"Final answer: {answer_item['answer']}")
                logger.info(f"Used premises: {answer_item['idx']}")
                logger.info("Answer item validated successfully")
            else:
                logger.error("Parsed JSON failed validation. Check model output format.")
                # Keep potentially incomplete data from JSON parse attempt

        else:
            logger.warning("Multi-step pipeline failed to produce valid JSON. Trying fallback pipeline...")
            # Fallback to original pipeline
            try:
                final_json_str, fallback_step_times = generate_pipeline(
                    premises=premise_dict,
                    question=question,
                    tokenizer=tokenizer,
                    model=model,
                    trace=True
                )
                # Add fallback timing information
                answer_item['timing']['fallback_used'] = True
                for key, value in fallback_step_times.items():
                    answer_item['timing'][f'fallback_{key}'] = value
                
                # Re-parse with fallback result
                response_json = parse_final_json(final_json_str)
                
                if response_json:
                    # Process fallback result
                    raw_idx = response_json.get('idx', [])
                    processed_idx = []
                    if isinstance(raw_idx, list):
                        for item in raw_idx:
                            try:
                                processed_idx.append(int(item))
                            except (ValueError, TypeError):
                                logger.warning(f"Non-integer value found in fallback idx list: {item}. Skipping.")
                    else:
                        logger.warning(f"Fallback parsed 'idx' is not a list ({type(raw_idx)}). Setting to [].")
                    
                    answer_item['answer'] = str(response_json.get('answer', 'Error: Missing Answer'))
                    answer_item['idx'] = sorted(list(set(processed_idx)))
                    answer_item['explanation'] = response_json.get('explanation', 'Error: Missing Explanation')
                    
                    if validate_answer_item(answer_item):
                        logger.info(f"Fallback final answer: {answer_item['answer']}")
                        logger.info(f"Fallback used premises: {answer_item['idx']}")
                        logger.info("Fallback answer item validated successfully")
                else:
                    # Both primary and fallback failed
                    logger.error("Both primary and fallback pipelines failed to produce valid JSON.")
                    # Attempt to extract indices from raw text
                    idx_matches = re.findall(r'premise\s+(\d+)', final_json_str.lower())
                    if idx_matches:
                        answer_item['idx'] = list(set([int(idx) for idx in idx_matches]))
                        logger.info(f"Extracted premise indices from raw response: {answer_item['idx']}")
                    
                    answer_item['answer'] = 'Error: Parse Failed'
                    answer_item['explanation'] = f"Failed to parse JSON. Raw Output: {final_json_str or '[No Raw Output]'}"
                    answer_item['timing']['error'] = 'Both pipelines failed'
            except Exception as e:
                logger.error(f"Fallback pipeline also failed: {str(e)}")
                # Original error handling
                if final_json_str:
                    # Attempt to extract indices even if JSON parsing failed
                    idx_matches = re.findall(r'premise\s+(\d+)', final_json_str.lower())
                    if idx_matches:
                        answer_item['idx'] = list(set([int(idx) for idx in idx_matches]))
                        logger.info(f"Extracted premise indices from raw response: {answer_item['idx']}")
                    
                    answer_item['answer'] = 'Error: Parse Failed'
                    answer_item['explanation'] = f"Failed to parse JSON. Raw Output: {final_json_str or '[No Raw Output]'}"
                    answer_item['timing']['error'] = 'Both pipelines failed with exceptions'
    except Exception as e:
        logger.error(f"Error processing sample: {str(e)}")
        if not answer_item['answer']:
            answer_item['answer'] = 'Unknown'
        if not answer_item['idx']:
            answer_item['idx'] = []
        if not answer_item['explanation']:
            answer_item['explanation'] = f"Error occurred: {str(e)}"
        if not answer_item['timing']:
            answer_item['timing'] = {'error': 'No timing information available'}

    if validate_answer_item(answer_item):
        logger.info("Answer item is complete and valid")
    else:
        logger.warning("WARNING: Answer item is incomplete or invalid")
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

    logger.info(f"\nSample {i-1} processing time: {sample_processing_time:.2f} seconds")
    logger.info(f"Average time per sample: {avg_time_per_sample:.2f} seconds")
    logger.info(f"Estimated remaining time: {estimated_remaining_time:.2f} seconds ({estimated_remaining_time/60:.2f} minutes)")

# Validate and save answers
logger.info("\nSaving results...")
save_start_time = time.time()
output = "/home/manh/Projects/QAChallenge/hm_answers.json"

valid_answers = []
for idx, answer_item in enumerate(answers):
    # Ensure basic structure even if validation failed during processing
    valid_answers.append({
        'question': answer_item.get('question', 'Missing Question'),
        'answer': answer_item.get('answer', 'Unknown'),
        'idx': answer_item.get('idx', []),
        'explanation': answer_item.get('explanation', 'No Explanation'),
        'timing': answer_item.get('timing', {'error': 'Timing N/A'})
    })

try:
    with open(output, "w", encoding="utf-8") as f:
        json.dump(valid_answers, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output}")
except Exception as e:
    logger.error(f"Failed to save results to {output}: {e}")

save_time = time.time() - save_start_time
total_time = time.time() - total_start_time

logger.info(f"Final save took {save_time:.2f} seconds")
logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
logger.info(f"Data processing time: {total_processing_time:.2f} seconds")
logger.info(f"Average time per sample: {total_processing_time/len(ds):.2f} seconds")