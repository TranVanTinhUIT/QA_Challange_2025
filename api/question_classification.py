import re

# PROMPT_SYS = """
# You are given a question.
# Your tasks:   
#   1. Classify the question into one or more relevant question types from the predefined list:
#     - 1 for Yes/No/Uncertain question.
#     - 2 for Multiple choice question. 
#     - 3 for How many question.
#     - 4 for chained question including multiple question components.
#   2. Each type has a unique ID, Output only the ID of the selected question type — no explanation, no text.
# """
PROMPT_SYS = """
You are an expert question classifier. Your task is to analyze and classify questions with absolute precision.

Given a question, classify it into EXACTLY ONE of these categories:

1. Yes/No/Uncertain (ID: 1)
   - Questions that can be answered with Yes, No, or Uncertain
   - Examples: "Is X true?", "Does Y exist?", "Can Z happen?", etc.
   - Must be a single, direct question requiring a yes/no/uncertain response

2. Multiple Choice (ID: 2)
   - Questions that present multiple options to choose from
   - Must explicitly list choices (A, B, C, D, etc.)
   - Examples: "Which of the following...", "Select the correct option...", etc.

3. How Many/Quantity (ID: 3)
   - Questions asking for a specific number or quantity
   - Must start with "How many", "What is the number of", etc.
   - Examples: "How many X are there?", "What is the count of Y?", etc.

4. Chained/Multi-part (ID: 4)
   - Questions containing multiple distinct questions that can be separated
   - Must have clear separators (e.g., "and", "or", ";")
   - Each sub-question must be classified into one of the above types (1, 2, or 3)
   - Examples: 
     * "Is X true and how many Y are there?" -> (1,3)
     * "Which option is correct and how many Z exist?" -> (2,3)
     * "Is A true or is B false?" -> (1,1)
     * "How many X are there and which of the following is correct?" -> (3,2)

Rules:
1. For single questions (IDs 1-3):
   - Output ONLY the numeric ID (1-3)
   - Choose the MOST SPECIFIC category that matches the question

2. For chained questions (ID: 4):
   - First output "4" to indicate it's a chained question
   - Then analyze each sub-question and output their types in order
   - Format: "4" followed by the sequence of sub-question types in parentheses
   - Example outputs:
     * "4(1,3)" for "Is X true and how many Y are there?"
     * "4(2,3)" for "Which option is correct and how many Z exist?"
     * "4(1,1)" for "Is A true or is B false?"
     * "4(3,2)" for "How many X are there and which of the following is correct?"

3. Important:
   - Each sub-question in a chained question must be classified as type 1, 2, or 3
   - Maintain the order of sub-questions in the output
   - Do not include any explanations or additional text
"""

PROMPT_USER = """
<?question>
"""

class QuestionClassification:

    def __init__(self):
        pass

    def classify(self, question, tokenizer, model, trace = False):
        """
        Return a list types of the question or `None` if can't detect.
        Supported types:
        - 1 for Yes/No/Uncertain
        - 2 for Multiple choice
        - 3 for How many.
        
        The ouput is id of question in a list:
         - [ 1 ] for Yes/No/Uncertain
         - [ 2 ] for Multiple choice
         - [ 3 ] for How many.
         - [ 3, 1 ] for Chained questions include How many + Yes/No/Uncertain

        """

        sys = PROMPT_SYS
        user = PROMPT_USER.replace('<?question>', question)

        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user }
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_ids = model_inputs.input_ids

        # Tạo toàn bộ kết quả với max_new_tokens lớn hơn
        outputs = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )

        # Decode output (including both prompt and generated text)
        generated_sequence = outputs.sequences[0]
        full_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        
        # Get only generated text (remove prompt)
        input_text_len = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))

        text = full_text[input_text_len:].strip()
        
        if trace:
            print('[CLASSIFICATION] output => ', text)

        # Find all occurrences of 1, 2, or 3
        matches = re.findall(r'\b[123]\b', text)

        # Return the last match if there are any
        types = matches if matches else None

        return types