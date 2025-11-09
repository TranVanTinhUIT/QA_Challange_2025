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
You are provided with a list of premises, which are statements expressed in natural language. Your task is to answer a given question based solely on these premises, without using any external information. The question may combine multiple types, such as Yes/No, Multiple Choice, How Many, or other forms, and may require reasoning across these types to arrive at a complete answer. You must:

Strictly adhere to the premises provided and avoid introducing external assumptions or knowledge.
Identify the components of the question (e.g., Yes/No, Multiple Choice, How Many) and address each part systematically.
Perform the reasoning step-by-step using a chain-of-thought approach, clearly explaining how each step connects to the premises and contributes to the final answer.
Ensure that all premises are carefully reviewed, and reasoning steps are interconnected and validated before proceeding.
For questions with Multiple Choice components, identify the correct option(s) and explain why other options are incorrect, if applicable.
For questions with quantitative components (e.g., "How Many"), provide a precise count or explanation based on the premises.
For questions with Yes/No components, provide a clear "Yes," "No," or "Uncertain" answer with justification.
If the question has mixed components, break it down into sub-questions, answer each part, and synthesize the results into a cohesive final answer.
If the premises are insufficient to answer any part of the question definitively, explain why and state any assumptions that would be needed, but do not use them to form the answer.
Conclude with a clear and concise final answer that addresses all components of the question.
Steps to follow:

Step 1: Restate the premises and the question clearly.
Step 2: Analyze the question to identify its components (e.g., Yes/No, Multiple Choice, How Many) and their relationships.
Step 3: Break down the question into sub-questions if it has mixed components, and map each to relevant premises.
Step 4: Perform step-by-step reasoning for each sub-question, linking each step to the premises and validating against prior steps.
Step 5: Synthesize the results of all sub-questions to form a cohesive answer.
Step 6: Summarize the reasoning and check for completeness across all question components.
Step 7: Provide the final answer in the format: Final Answer: followed by the response, addressing all parts of the question.
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
You are provided with three pieces of information:

Premises: A list of statements expressed in natural language, indexed starting from 0.
Question: A question to be answered, which may consist of multiple parts or combine different types (e.g., Yes/No, Multiple Choice, How Many, or others).
Reasoning: The reasoning process and the answer(s) to the question, which must be strictly followed.
Your task is to:

Strictly adhere to the provided reasoning, ensuring consistency with it, and use it to derive the answer(s).
Answer all parts of the question thoroughly and completely, addressing each component explicitly, whether the question is a single type (e.g., Yes/No) or a mix of types (e.g., Yes/No + Multiple Choice + How Many).
Validate that the reasoning covers all parts of the question. If the reasoning is incomplete (i.e., does not address all parts), identify the missing parts and provide a clear explanation in the explanation field, but do not introduce external information or assumptions to fill the gaps.
Produce the output in the exact format:
json

{
    "answer": "",
    "idx": [],
    "explanation": ""
}
Where:
answer: The complete answer to the question, addressing all parts.
For Yes/No parts, use "Yes," "No," or "Uncertain."
For Multiple Choice parts, use the correct option(s) (e.g., "A", "B", or "A, B").
For How Many or calculation-based parts, use a number (e.g., "2").
For mixed questions, combine answers for all parts in a clear, concise format (e.g., "Yes, 2, B").
idx: A list of indices (integers) of the premises used in the reasoning, consistent with the provided reasoning. Include only the indices explicitly referenced or implied in the reasoning.
explanation: A clear, concise explanation of the reasoning process, referencing the premises by their indices (e.g., "Premise[0]") and explaining how they lead to the answer for each part of the question. If the reasoning is incomplete for any part, note this explicitly and explain what is missing.
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
