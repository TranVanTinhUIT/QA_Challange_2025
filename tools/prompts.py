# import json

# with open('prompts.json', 'r') as file:
#     prompts = json.load(file)

# guideline_prompt = prompts["guideline_prompt"]
# which_type_prompt = prompts["which_type_prompt"]
# pre_prompt = prompts["pre_prompt"]
# pos_prompt = prompts["pos_prompt"]
# bi_prompt  = prompts["bi_prompt"]
# mul_prompt = prompts["mul_prompt"]
# hm_prompt  = prompts["hm_prompt"]



guideline_prompt = """You are my assistant. Your task is to carefully read and strictly follow the instructions I provide to answer my query. My input consists of two parts:
1. Instructions: This section outlines specific guidelines for responding to my query.
2. Query: This section contains the query I need you to answer.
You must adhere strictly to the guidelines in the Instructions section without any deviation. Ensure your responses are accurate, relevant, and aligned with the provided instructions."""

which_type_prompt = """You are tasked with classifying the question I provide into one of three categories, regardless of any missing or unclear information. The categories are:
1. Yes/No question: Output \"1\" if the question requires a yes or no answer.
2. How many question: Output \"2\" if the question asks for a numerical quantity or count.
3. Multiple choice question: Output \"3\" if the question provides multiple answer options.
Strictly follow these guidelines and output only the corresponding number (1, 2, or 3) based on the question's type."""

pre_prompt = """Before proceeding to the question, I will provide you with premises, which are statements expressed in natural language. Each premise is prefixed with an index, for example:
Premise 1: If a Python code is well-tested, then the project is optimized.
Premise 2: If a Python code does not follow PEP 8 standards, then it is not well-tested.
You must answer the question based on these premises, strictly adhering to them. You must provide the response in the following JSON format: {\"answer\": \"your answer here\", \"idx\": [list of premise indexes used], \"explanation\": \"your explanation based on the premises\"}. In which:\n"""

pos_prompt = """\"explanation\": Provide a clear and concise explanation for your answer, detailing the reasoning process. You must reference the premises used by their indexes (e.g., Premise 1, Premise 2) in the explanation. Ensure all referenced premises are clearly identified by their indexes.
\"idx\": A list of integers (e.g., [1, 2]) representing the indexes of the premises used in the explanation. Ensure strict consistency: every index in idx must correspond to a premise referenced in the explanation, and every premise referenced in the explanation must have its index included in idx.\n"""

bi_prompt  = """{}\"answer\": Because this is a Yes/No/Uncertain question, you should output one of the following based on the given premises:
1. \"Yes\" if the question can be proven true.
2. \"No\" if the question can be proven false.
3. \"Uncertain\" if the question cannot be proven true or false, meaning no conclusion can be drawn.\n{}""".format(pre_prompt, pos_prompt)

mul_prompt = """{}\"answer\": Because this is an multiple-choice question, you should output the correct option (e.g., \"A\", \"B\", \"C\", \"D\") based on the question and the given premises.\n{}""".format(pre_prompt, pos_prompt)

hm_prompt  = """{}\"answer\": Because this is a how many question, you need to output a single integer based on the question and the given premises.\n{}""".format(pre_prompt, pos_prompt)

math_instruction_prompt = """
You are provided with a list of premises, which are statements expressed in natural language. Your task is to answer the given question based solely on these premises, without using any external information. You must strictly adhere to the premises and perform the reasoning step-by-step using a chain-of-thought approach to arrive at the final answer. The reasoning steps should be clear, logical, and presented before stating the conclusion. Ensuring that you carefully review all the premises and previous reasoning steps before proceeding, as the premises are interconnected and form the basis for the reasoning. If you believe the prior reasoning steps are sufficient to form the answer, provide the final answer by starting with "Final Answer:" followed by the response.
"""

math_prompt = """
Now, I will provide you with the premises and question:

Premises:
{}

Question:
{}

That is all the information. Now it is your turn to respond.
"""

reasoning_instructions_prompt="""
Instructions:
I will provide you with three pieces of information:

1. Premises: These are statements expressed in natural language.
2. Question: This is the question to be answered.
3. Reasoning: This is the reasoning process and the answer to the question.
Your task is to strictly adhere to the provided reasoning (ensuring consistency with it) and produce an output in the following format:
{
  "answer": "",
  "idx": [],
  "explanation": ""
}
Where:
{
    answer: The answer to the question. If the question is of the Yes/No type, you must answer with Yes, No, or Uncertain. If the question is multiple choice, output the correct answer (A, B, C, D, etc.). If the question involves calculations or numbers, output the number. If the question is of the "how many" type, also output a number. In short, respond directly to the question without being wordy. Moreover, if the question is a mix of the types above, answer each part accordingly.
    idx: A list of indices corresponding to the premises used in the reasoning (must be consistent with the reasoning).
    explanation: An explanation of the reasoning process, clearly referencing the premises (using their indices) that were used to arrive at the answer.
}
"""

reasoning_prompt = """
Now, I will provide you with the necessary information to respond:

Premises:
{}

Question:
{}

Reasoning:
{}

That is all the information. Now it is your turn to respond.
"""
