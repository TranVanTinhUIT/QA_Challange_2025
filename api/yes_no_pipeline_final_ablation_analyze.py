import torch
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import re
import time
from datetime import datetime
# Make imports conditional if bitsandbytes might be missing
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer
except ImportError:
    print("Warning: transformers library not found or bitsandbytes/accelerate missing. Quantization may not be available.")
    # Define dummy types if library is missing, to avoid NameErrors
    PreTrainedModel = type('PreTrainedModel', (object,), {})
    PreTrainedTokenizer = type('PreTrainedTokenizer', (object,), {})
    # Fallback if Auto* classes are needed elsewhere, though they shouldn't be directly used if import fails
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None # type: ignore


# --- Prompts Definitions ---
# (Keep these accessible, either as module-level constants or class attributes)

PROMPT_COT_SYS_YN = """
You are a helpful assistant trained to answer the question based on the given domain knowledge. Given premises and a Yes/No/Uncertain question, reason step-by-step  to answer the question.

**Strict Requirements**
 - Using ONLY the minimal set of given premises necessary to answer.
 - Avoid including premises that are not directly required to derive the answer.
 - Focus on logical deduction and clearly state which premises support each step.
 - If the given premises do not contain sufficient information to answer the question with certainty, 
    your answer should be 'Uncertain' with an empty list of premise indices. 
"""

PROMPT_COT_USER_STEP_1 = """
## Premises:
<?premises>

## Question: <?question>

Reasoning Steps Towards Answer:
(Provide a focused, step-by-step deduction using the premises to answer the question, enclosing the reasoning in response within <reasoning></reasoning> tags.)
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
-   If no premises were cited, output nothing or an empty line.
-   If the answer is 'Uncertain' because the premises are insufficient, leave this empty.
"""

QS_TYPE_YN = 'Yes/No/Uncertain'

# Default generation parameters (can be overridden in process method if needed)
ANALYZE_TOKENS_PER_PREMISE = 70
DEFAULT_MAX_TOKENS_STEP_1 = 650
DEFAULT_MAX_TOKENS_STEP_2 = 30
DEFAULT_MAX_TOKENS_STEP_3 = 400
DEFAULT_MAX_TOKENS_STEP_4 = 50
DEFAULT_MAX_TOKENS_STEP_5 = 500
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

class YesNoPipeline:
    """
    Encapsulates the multi-step logic for answering Yes/No/Uncertain questions based on premises.

    call `run` method to run the pipeline for a given question and premises.

    """
    def __init__(self):
        pass
    
    # --- Private Helper Methods --- 

    def _generate_prompt(self, template: str, data_dict: Dict[str, Any]) -> str:
        """Helper to fill prompt templates safely."""
        prompt = template
        for key, val in data_dict.items():
            target = f'<?{key}>'
            if target in prompt:
                prompt = prompt.replace(target, str(val))
        return prompt

    def _create_sys_prompt(self) -> str:
        """Creates the system prompt."""
        return self._generate_prompt(PROMPT_COT_SYS_YN, data_dict={})

    def _create_user_prompt_step1(self, premises: List[str], question: str) -> str:
        """Creates the user prompt for the initial reasoning step."""
        data_dict = {}
        premise_vals = ""
        for i, premise in enumerate(premises, 1):
            premise_vals += f"    - Premise {i}: {premise}\n"
        data_dict['premises'] = premise_vals
        data_dict['question'] = question
        return self._generate_prompt(PROMPT_COT_USER_STEP_1, data_dict=data_dict)

    def _create_user_prompt_step2(self) -> str:
        """Creates the user prompt for extracting the concise answer."""
        return self._generate_prompt(PROMPT_COT_USER_STEP_2_YN, {})

    def _create_user_prompt_step3(self, concise_answer: str) -> str:
        """Creates the user prompt for generating the explanation."""
        return self._generate_prompt(PROMPT_COT_USER_STEP_3_EXPLANATION, {"concise_answer": concise_answer})

    def _create_user_prompt_step4(self, explanation: str) -> str:
        """Creates the user prompt for extracting premise indices."""
        return self._generate_prompt(PROMPT_COT_USER_STEP_4_IDX, {"explanation": explanation})

    @torch.no_grad() # Disable gradient calculations for inference
    def _generate_answer(self, messages: List[Dict[str, str]], tokenizer, model, max_tokens: int) -> str:
        """Generates a response from the model, handling potential errors."""
        # # Check for initialized model/tokenizer
        # if not self.model or not self.tokenizer:
        #     print("Error: Pipeline not properly initialized during generation call.")
        #     # Return an error string that the calling function can check
        #     return "Error: Pipeline not initialized."

        # Validate message structure
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            print("Error: 'messages' must be a list of dictionaries.")
            return "Error: Invalid message format"
        valid_messages = [m for m in messages if isinstance(m.get("content"), str) and m["content"]]
        if not valid_messages:
            print("Error: No valid messages with non-empty string content to process.")
            return "Error: No valid messages"

        # Prepare input text
        try:
            text = tokenizer.apply_chat_template(
                valid_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Error applying chat template: {e}")
            return f"Error: Applying chat template failed - {e}"

        # Generate response
        try:
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            input_ids = model_inputs.input_ids

            # Determine EOS token ID
            eos_token_id_to_use = tokenizer.eos_token_id
            if isinstance(eos_token_id_to_use, list):
                eos_token_id_to_use = eos_token_id_to_use[0]
            if eos_token_id_to_use is None:
                eos_token_id_to_use = tokenizer.pad_token_id
                if eos_token_id_to_use is None:
                    print("Critical Error: Neither EOS nor PAD token ID is set.")
                    return "Error: Missing EOS/PAD token ID"
                else:
                    print("Warning: Using PAD token ID as EOS token ID.")

            # Perform generation
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id_to_use,
                return_dict_in_generate=True,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P
            )

            # Decode and extract generated part
            generated_sequence = output.sequences[0]
            full_generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            input_text_len = len(decoded_input)
            generated_part = full_generated_text[input_text_len:].strip()

            # Attempt to strip ```json markdown if present
            match = re.search(r"^```json\s*(\{[\s\S]*?\})\s*```$", generated_part, re.DOTALL)
            if match:
                return match.group(1).strip()
            # Return the potentially plain JSON or other text otherwise
            return generated_part

        except Exception as e:
            # Catch errors during tokenization, generation, or decoding
            print(f"Error during model generation or decoding: {e}")
            import traceback
            traceback.print_exc()
            # Return an error string
            return f"Error: Generation/Decoding failed - {e}"

    def _parse_final_output(self, final_output_str: str, sample_info: str = "") -> Optional[Dict[str, Any]]:
        """
        Attempts to parse the final string output (expected JSON) from the LLM,
        validates structure and content, and returns a cleaned dictionary or None.
        """
        print(f"Attempting to parse final output {sample_info}...")

        # Check for None or empty string, or if it doesn't look like JSON
        if not final_output_str or not final_output_str.strip().startswith('{'):
            print(f"Final output is empty or doesn't start with '{{'. Raw: '{final_output_str}'")
            return None

        try:
            # Attempt to load the string directly as JSON
            parsed_json = json.loads(final_output_str)
            print(f"Successfully parsed potential JSON {sample_info}.")

            # --- Structure Validation ---
            if not isinstance(parsed_json, dict):
                print(f"Error {sample_info}: Parsed result is not a dictionary (type: {type(parsed_json)}).")
                return None
            required_keys = {'answer', 'idx', 'explanation'}
            if not required_keys.issubset(parsed_json.keys()):
                missing_keys = required_keys - set(parsed_json.keys())
                print(f"Error {sample_info}: Parsed JSON missing required keys: {missing_keys}. Parsed: {parsed_json}")
                return None

            # --- Field Value Validation and Cleaning ---
            answer = parsed_json.get('answer')
            idx_raw = parsed_json.get('idx') # Get raw value first
            explanation = parsed_json.get('explanation')

            # Validate Answer
            if not isinstance(answer, str) or answer not in ['Yes', 'No', 'Uncertain']:
                print(f"Error {sample_info}: Invalid or non-string answer value '{answer}'.")
                return None

            # Validate and Clean Index List
            idx_cleaned = []
            if not isinstance(idx_raw, list):
                print(f"Warning {sample_info}: Parsed 'idx' is not a list ({type(idx_raw)}). Using empty list.")
            else:
                try:
                    # Convert valid items to int, ensure uniqueness, sort
                    valid_ints = {int(x) for x in idx_raw if isinstance(x, (int, str)) and str(x).isdigit()}
                    idx_cleaned = sorted(list(valid_ints))
                except (ValueError, TypeError) as conv_err:
                    # Log conversion error but proceed with empty list
                    print(f"Warning {sample_info}: Error converting idx items to int ({conv_err}). Raw idx: {idx_raw}. Using empty list.")

            # Validate Explanation
            if not isinstance(explanation, str) or not explanation:
                print(f"Error {sample_info}: Explanation is missing, not a string, or empty.")
                return None

            # --- Consistency Validation ---
            if answer == 'Uncertain' and idx_cleaned:
                print(f"Info {sample_info}: Correcting idx to [] for Uncertain answer (was {idx_cleaned}).")
                idx_cleaned = []

            # Return the cleaned and validated dictionary
            print(f"Parsing and validation successful {sample_info}.")
            return {
                "answer": answer,
                "idx": idx_cleaned,
                "explanation": explanation # Return the explanation as parsed
            }

        except json.JSONDecodeError as json_err:
            print(f"JSON parsing failed {sample_info}: {json_err}")
            print(f"Raw output string causing error:\n---\n{final_output_str}\n---")
            return None
        except Exception as proc_err:
            # Catch other errors during validation/cleaning
            print(f"An unexpected error occurred during JSON processing {sample_info}: {proc_err}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_reasoning_step_1(self, llm_response):
        # entities_start = llm_response.find('<entities>')
        # entities_end = llm_response.find('</entities>')
        # entities = ''
        # if entities_start > 0 and entities_end > 0 and entities_start < entities_end :
        #     entities = llm_response[entities_start: entities_end].replace('<entities>', '').replace('</entities>', '').strip()

        reasoning_start = llm_response.find('<reasoning>')
        reasoning_end = llm_response.find('</reasoning>')
        reasoning = ''
        if reasoning_start >= 0 and reasoning_end >= 0 and reasoning_start < reasoning_end :
            reasoning = llm_response[reasoning_start: reasoning_end].replace('<reasoning>', '').replace('</reasoning>', '').strip()
        return reasoning

    # --- Public Process Method --- 

    def run(self, premises: List[str], question: str, tokenizer, model, trace: bool = False) -> Dict[str, Any]:
        """
        Processes a single question with its premises using the multi-step pipeline.

        Args:
            premises: A list of premise strings.
            question: The question string.
            trace: If True, prints detailed step timing and intermediate information.

        Returns:
            A dictionary containing the processing results. It will always include:
            - 'question' (str): The original question.
            - 'answer' (str): 'Yes', 'No', 'Uncertain', or an error message.
            - 'idx' (List[int]): List of premise indices used (empty if Uncertain or error).
            - 'explanation' (str): Step-by-step explanation or error details.
            - 'timing' (Dict[str, float]): Execution time for each step and total.
            It may also include these keys in case of errors:
            - 'error' (str): A description of the error.
            - 'raw_output' (str): The raw output from the final LLM step, if parsing failed.
            - 'raw_output_before_error' (str): Raw output from the last successful step before an exception.
        """
        pipeline_start_time = time.time()
        step_times: Dict[str, float] = {}
        final_output_str: Optional[str] = None # Store raw output from final step
        result_dict: Dict[str, Any] = { # Initialize consistent result structure
            "question": question,
            "answer": "Error: Pipeline Incomplete",
            "idx": [],
            "explanation": "Pipeline did not complete successfully.",
            "timing": step_times, # Reference the timing dict
            "error": None, # Add error key, initially None,
            'entities': '',
            'reasoning': ''
        }

        try:
            # --- Input Validation ---
            if not isinstance(premises, list) or not premises:
                raise ValueError("Input 'premises' must be a non-empty list of strings.")
            if not isinstance(question, str) or not question:
                raise ValueError("Input 'question' must be a non-empty string.")

            # --- Step 1: Initial Reasoning ---            
            sys_prompt = self._create_sys_prompt()
            user_prompt_s1 = self._create_user_prompt_step1(premises, question)
            messages_s1 = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt_s1}]
            if trace: print('Step 1: Generating initial reasoning...')
            s1_start = time.time()
            step_1_reasoning = self._generate_answer(messages=messages_s1, tokenizer=tokenizer, model=model, max_tokens= DEFAULT_MAX_TOKENS_STEP_1)
            if trace: print('Step 1 => \n', step_1_reasoning)
            
            step_times['step_1_reasoning'] = time.time() - s1_start
            if trace: print(f"Step 1 took {step_times['step_1_reasoning']:.2f}s")
            if step_1_reasoning.startswith("Error:"): raise RuntimeError(f"Step 1 failed: {step_1_reasoning}")
            final_output_str = step_1_reasoning # Store last successful output

            reasoning = self.extract_reasoning_step_1(step_1_reasoning)
            # result_dict['entities'] = entities
            result_dict['reasoning'] = reasoning
            if trace: print('Extract reasoning => \n', reasoning)

            # --- Step 2: Concise Answer ---            
            user_prompt_s2 = self._create_user_prompt_step2()
            messages_s2 = messages_s1 + [{"role": "system", "content": step_1_reasoning}, {"role": "user", "content": user_prompt_s2}] 
            if trace: print('Step 2: Extracting concise answer...')
            s2_start = time.time()
            step_2_raw = self._generate_answer(messages=messages_s2, tokenizer=tokenizer, model=model, max_tokens= DEFAULT_MAX_TOKENS_STEP_2)
            match = re.search(r'\b(Yes|No|Uncertain)\b', step_2_raw, re.IGNORECASE)
            step_2_answer = match.group(1).capitalize() if match else "Uncertain"
            step_times['step_2_concise_answer'] = time.time() - s2_start
            if trace: print(f"Step 2 took {step_times['step_2_concise_answer']:.2f}s. Answer: {step_2_answer}")
            if step_2_raw.startswith("Error:"): raise RuntimeError(f"Step 2 failed: {step_2_raw}")
            final_output_str = step_2_raw

            # --- Step 3: Explanation ---            
            user_prompt_s3 = self._create_user_prompt_step3(step_2_answer)
            messages_s3 = messages_s1 + [ 
                {"role": "system", "content": step_1_reasoning},
                {"role": "system", "content": f"Concise Answer: {step_2_answer}"}, 
                {"role": "user", "content": user_prompt_s3}
            ]
            if trace: print('Step 3: Generating concise explanation...')
            s3_start = time.time()
            step_3_explanation = self._generate_answer(messages=messages_s3, tokenizer=tokenizer, model=model, max_tokens= DEFAULT_MAX_TOKENS_STEP_3)
            step_times['step_3_explanation'] = time.time() - s3_start
            if trace: print(f"Step 3 took {step_times['step_3_explanation']:.2f}s")
            if step_3_explanation.startswith("Error:"): raise RuntimeError(f"Step 3 failed: {step_3_explanation}")
            final_output_str = step_3_explanation

            # --- Step 4: Indices ---            
            step_4_indices_str = "" # Default to empty string
            if step_2_answer != "Uncertain":             
                user_prompt_s4 = self._create_user_prompt_step4(step_3_explanation)
                messages_s4 = messages_s3 + [ 
                    {"role": "system", "content": step_3_explanation}, 
                    {"role": "user", "content": user_prompt_s4}
                ] 
                if trace: print('Step 4: Extracting premise indices...')
                s4_start = time.time()
                step_4_raw = self._generate_answer(messages=messages_s4,tokenizer=tokenizer, model=model, max_tokens= DEFAULT_MAX_TOKENS_STEP_4)
                step_4_indices_str = ",".join(re.findall(r'\b(\d+)\b', step_4_raw))
                step_times['step_4_indices'] = time.time() - s4_start
                if trace: print(f"Step 4 took {step_times['step_4_indices']:.2f}s. Indices: '{step_4_indices_str}'")
                if step_4_raw.startswith("Error:"): raise RuntimeError(f"Step 4 failed: {step_4_raw}")
                final_output_str = step_4_raw
            else:
                if trace: print("Step 4: Skipped index extraction (Answer is Uncertain)")
                step_times['step_4_indices'] = 0.0 # Indicate skipped step

            # --- Step 5: Synthetic final Response ---
            result_dict['answer'] = step_2_answer
            idx = []
            try:
                if step_4_indices_str and step_4_indices_str.strip():
                    arr = [int(i.strip()) for i in step_4_indices_str.split(',') if i.strip().isdigit()]
                    # Ensure indices are unique and sorted before creating JSON list string
                    idx = sorted(list(set(arr)))

            except Exception as e:
                # ignore
                idx = []

            result_dict['idx'] = idx
            result_dict['explanation'] = step_3_explanation

        except ValueError as ve:
            # Handle input validation errors
            print(f"Input validation error: {ve}")
            result_dict['answer'] = "Error: Invalid Input"
            result_dict['explanation'] = str(ve)
            result_dict['error'] = f"Input validation failed: {ve}"
        except Exception as e:
            # Handle errors during pipeline execution (e.g., generation failures)
            import traceback
            print(f"Error during pipeline processing: {e}")
            traceback.print_exc()
            result_dict['answer'] = "Error: Pipeline Exception"
            result_dict['explanation'] = f"Pipeline Error: {e}"
            result_dict['error'] = f"Pipeline execution failed: {e}"
            # Store raw output from the step that likely failed
            result_dict['raw_output_before_error'] = final_output_str
        
        # Final timing calculation
        total_time = time.time() - pipeline_start_time
        step_times['total_pipeline'] = total_time
        result_dict['timing'] = step_times # Ensure timing is always included
        if trace: print(f"Total pipeline time: {total_time:.2f}s")
        
        return result_dict


def get_model(model_name = "Qwen/Qwen2.5-32B-Instruct-AWQ"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model

if __name__ == "__main__":
    pipeline = YesNoPipeline()
    tokenizer, model = get_model() # Model

    ds_folder = os.path.dirname(os.path.abspath(__file__)) + '/../dataset'
    out_folder = os.path.dirname(os.path.abspath(__file__)) + '/../out'

    with open(ds_folder + '/yn_test.json', "r", encoding="utf-8") as f:
        yn_test = json.load(f)
    
    test_ds = yn_test
    print(len(test_ds))

    results = []

    out_file = out_folder + f"/yes_no_pipeline_ablation_analyze.json"
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    
    for index, item in enumerate(test_ds):
        if index < len(results):
            continue
        print('Test item ', index)
        if index %10 == 0 :
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        premises = item['premises-NL']
        question = item['question']
        start_time = time.perf_counter()
        yn_result = pipeline.run(premises=premises, question=question, tokenizer=tokenizer, model=model, trace=True)
        endtime_time = time.perf_counter()
        result = {
            'premises': premises,
            'question': question
        }
        result['time'] = endtime_time - start_time
        
        result['ref_answer'] = item['answer']
        result['ref_index'] = item['idx']
        result['ref_explanation'] = item['explanation']
        result['pred_answer'] = yn_result['answer']
        result['pred_idx'] = yn_result['idx']
        result['pred_explanation'] = yn_result['explanation']
        result['entities'] = yn_result['entities']
        result['reasoning'] = yn_result['reasoning']
        results.append(result)
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
   