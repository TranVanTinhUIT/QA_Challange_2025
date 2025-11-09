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
-   If no premises were cited, output nothing or an empty line.
-   If the answer is 'Uncertain' because the premises are insufficient, leave this empty.
"""

# Final prompt asks LLM to construct the JSON
PROMPT_COT_USER_FINAL_YN = """
Now, based on the following information:
- Concise Answer: '<?concise_answer>'
- Identified Premise Indices: <?idx_list_str> # String representation, e.g., "1,3,5" or ""
- Step-by-Step Explanation: "<?explanation>"

Construct the final JSON object EXACTLY in the format below. Output ONLY the JSON object and nothing else.

Response in the following JSON format:
```json
{
  "answer": "<Yes | No | Uncertain>",
  "idx": [<List the index of used premises>],
  "explanation": "<explanation>"
}
where:
  - `answer` is the answer to the question.
  - `explanation` is detailed explanation which is escaped json. Proper escaping is crucial to prevent JSON syntax errors — DO NOT VIOLATE it.
  - `idx` is list the index of premises referenced in explanation.
```

"""

QS_TYPE_YN = 'Yes/No/Uncertain'

# Default generation parameters (can be overridden in process method if needed)
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

    Loads a specified Hugging Face model and tokenizer on initialization and provides
    a `process` method to run the pipeline for a given question and premises.

    Attributes:
        model_name (str): Name of the loaded Hugging Face model.
        tokenizer (PreTrainedTokenizer): The loaded tokenizer.
        model (PreTrainedModel): The loaded language model.
        device (str): The device map used for the model.
    """
    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 use_quantization: bool = True,
                 device_map: str = "auto",
                 torch_dtype: Any = "auto", # Use Any for flexibility, common default is "auto"
                 trust_remote_code: bool = True):
        """
        Initializes the pipeline by loading the model and tokenizer.

        Args:
            model_name: The name or path of the Hugging Face model to load.
            use_quantization: Whether to use BitsAndBytes 8-bit quantization.
                              Requires `bitsandbytes` and `accelerate` libraries.
            device_map: The device map strategy for model loading (e.g., "auto", "cuda:0").
            torch_dtype: The torch dtype for model loading (e.g., torch.float16, "auto").
            trust_remote_code: Whether to trust remote code execution for model loading.

        Raises:
            ImportError: If required libraries (transformers, torch, bitsandbytes) are missing.
            RuntimeError: If model/tokenizer loading fails for other reasons.
        """
        print(f"Initializing YesNoPipeline with model: {model_name}")
        # Check if necessary classes were imported
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("Transformers library not found. Please install it: pip install transformers")
        if use_quantization and BitsAndBytesConfig is None:
            raise ImportError("BitsAndBytesConfig not found. Please install bitsandbytes: pip install bitsandbytes")

        self.model_name = model_name
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.device = device_map # Store device preference for potential future use

        load_start_time = time.time()
        try:
            quantization_config = None
            if use_quantization:
                print("Applying 8-bit quantization...")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
            else:
                print("Quantization disabled.")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            print("Tokenizer loaded.")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                quantization_config=quantization_config,
                trust_remote_code=trust_remote_code,
            )
            print(f"Model loaded to device: {self.model.device}")

            if self.tokenizer.pad_token is None:
                # Common practice for models without a pad token
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("Set pad_token to eos_token")

            self.model.eval()  # Set model to evaluation mode
            print(f"Model and tokenizer initialized successfully in {time.time() - load_start_time:.2f} seconds.")

        except ImportError as ie:
            # Catch specific ImportErrors related to optional dependencies like bitsandbytes
            print(f"ImportError during initialization: {ie}. Please ensure necessary libraries are installed for the selected configuration (e.g., `pip install bitsandbytes accelerate` for quantization).")
            raise
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            # Wrap generic exceptions in RuntimeError for clarity
            raise RuntimeError(f"Failed to initialize YesNoPipeline: {e}") from e

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

    def _create_user_prompt_step5(self, concise_answer: str, idx_list_str: str, explanation: str) -> str:
        """Creates the user prompt for the final JSON generation step."""
        try:
            if idx_list_str and idx_list_str.strip():
                parsed_idx = [int(i.strip()) for i in idx_list_str.split(',') if i.strip().isdigit()]
                # Ensure indices are unique and sorted before creating JSON list string
                idx_list_json = json.dumps(sorted(list(set(parsed_idx))))
            else:
                idx_list_json = "[]"
        except Exception as e:
            print(f"Warning: Error converting index list string '{idx_list_str}' to JSON list: {e}")
            idx_list_json = "[]"

        # Escape the explanation string to be safely embedded within the JSON structure of the prompt
        # json.dumps adds quotes, so we slice them off [1:-1] if embedding inside existing quotes
        cleaned_explanation_escaped = json.dumps(explanation.strip())[1:-1]

        return self._generate_prompt(PROMPT_COT_USER_FINAL_YN, {
            "concise_answer": concise_answer,
            "idx_list_str": idx_list_str,
            "idx_list_json": idx_list_json,
            "explanation": cleaned_explanation_escaped
        })

    @torch.no_grad() # Disable gradient calculations for inference
    def _generate_answer(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        """Generates a response from the model, handling potential errors."""
        # Check for initialized model/tokenizer
        if not self.model or not self.tokenizer:
            print("Error: Pipeline not properly initialized during generation call.")
            # Return an error string that the calling function can check
            return "Error: Pipeline not initialized."

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
            text = self.tokenizer.apply_chat_template(
                valid_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Error applying chat template: {e}")
            return f"Error: Applying chat template failed - {e}"

        # Generate response
        try:
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_ids = model_inputs.input_ids

            # Determine EOS token ID
            eos_token_id_to_use = self.tokenizer.eos_token_id
            if isinstance(eos_token_id_to_use, list):
                eos_token_id_to_use = eos_token_id_to_use[0]
            if eos_token_id_to_use is None:
                eos_token_id_to_use = self.tokenizer.pad_token_id
                if eos_token_id_to_use is None:
                    print("Critical Error: Neither EOS nor PAD token ID is set.")
                    return "Error: Missing EOS/PAD token ID"
                else:
                    print("Warning: Using PAD token ID as EOS token ID.")

            # Perform generation
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id_to_use,
                return_dict_in_generate=True,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P
            )

            # Decode and extract generated part
            generated_sequence = output.sequences[0]
            full_generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
            decoded_input = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
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

    # --- Public Process Method --- 

    def process(self, premises: List[str], question: str, trace: bool = False) -> Dict[str, Any]:
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
            "error": None # Add error key, initially None
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
            step_1_reasoning = self._generate_answer(messages_s1, DEFAULT_MAX_TOKENS_STEP_1)
            step_times['step_1_reasoning'] = time.time() - s1_start
            if trace: print(f"Step 1 took {step_times['step_1_reasoning']:.2f}s")
            if step_1_reasoning.startswith("Error:"): raise RuntimeError(f"Step 1 failed: {step_1_reasoning}")
            final_output_str = step_1_reasoning # Store last successful output

            # --- Step 2: Concise Answer ---            
            user_prompt_s2 = self._create_user_prompt_step2()
            messages_s2 = messages_s1 + [{"role": "system", "content": step_1_reasoning}, {"role": "user", "content": user_prompt_s2}] 
            if trace: print('Step 2: Extracting concise answer...')
            s2_start = time.time()
            step_2_raw = self._generate_answer(messages_s2, DEFAULT_MAX_TOKENS_STEP_2)
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
            step_3_explanation = self._generate_answer(messages_s3, DEFAULT_MAX_TOKENS_STEP_3)
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
                step_4_raw = self._generate_answer(messages_s4, DEFAULT_MAX_TOKENS_STEP_4)
                step_4_indices_str = ",".join(re.findall(r'\b(\d+)\b', step_4_raw))
                step_times['step_4_indices'] = time.time() - s4_start
                if trace: print(f"Step 4 took {step_times['step_4_indices']:.2f}s. Indices: '{step_4_indices_str}'")
                if step_4_raw.startswith("Error:"): raise RuntimeError(f"Step 4 failed: {step_4_raw}")
                final_output_str = step_4_raw
            else:
                if trace: print("Step 4: Skipped index extraction (Answer is Uncertain)")
                step_times['step_4_indices'] = 0.0 # Indicate skipped step

            # --- Step 5: Final JSON Generation ---            
            user_prompt_s5 = self._create_user_prompt_step5(step_2_answer, step_4_indices_str, step_3_explanation)
            messages_s5 = messages_s1 + [ # Base context + final instruction            
                {"role": "system", "content": f"Context for JSON Generation:\nConcise Answer: {step_2_answer}\nIndices String: {step_4_indices_str}\nExplanation:\n{step_3_explanation}"}, 
                {"role": "user", "content": user_prompt_s5}
            ] 
            if trace: print('Step 5: Generating final JSON output...')
            s5_start = time.time()
            final_output_str = self._generate_answer(messages_s5, DEFAULT_MAX_TOKENS_STEP_5)
            step_times['final_json_generation'] = time.time() - s5_start
            if trace: print(f"Step 5 took {step_times['final_json_generation']:.2f}s")
            if final_output_str.startswith("Error:"): raise RuntimeError(f"Step 5 failed: {final_output_str}")

            # --- Parse Final Output ---            
            parsed_result = self._parse_final_output(final_output_str, sample_info=f"for question '{question[:50]}...'")

            # --- Populate Result Dictionary --- 
            if parsed_result:
                # Update result dict with parsed values
                result_dict['answer'] = parsed_result['answer']
                result_dict['idx'] = parsed_result['idx']
                result_dict['explanation'] = parsed_result['explanation']
                result_dict['error'] = None # Explicitly set error to None on success

                explanation = result_dict['explanation']

                import codecs
                explanation = codecs.decode(explanation, 'unicode_escape')
                explanation = explanation.replace('\n', ' ')
                explanation = explanation.replace('“', '"').replace('”', '"')  
                explanation = explanation.replace('’', "'")  
                explanation = explanation.replace('→', '→')  

                explanation = ' '.join(explanation.split())

                result_dict['explanation'] = explanation  

            else:
                # Parsing failed, update result dict with error info
                #result_dict['answer'] = "Error: Parse Failed"
                #result_dict['explanation'] = f"Parsing Failed. Raw Output: {final_output_str or '[No Raw Output]'}"
                #result_dict['error'] = "Failed to parse final JSON output from LLM."
                #result_dict['raw_output'] = final_output_str # Include raw output for debugging

                fixed_str = final_output_str + '"}'
                cleaned_str = re.sub(r'^```json\n', '', fixed_str)
                data = json.loads(cleaned_str)

                result_dict['answer'] = repr(data['answer'])
                result_dict['idx'] = data['idx']
                explanation = data['explanation']

                import codecs
                explanation = codecs.decode(explanation, 'unicode_escape')
                explanation = explanation.replace('\n', ' ')
                explanation = explanation.replace('“', '"').replace('”', '"')  
                explanation = explanation.replace('’', "'")  
                explanation = explanation.replace('→', '→')  

                explanation = ' '.join(explanation.split())

                result_dict['explanation'] = explanation  

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
