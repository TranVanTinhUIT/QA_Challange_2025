import json
import torch
import logging
from utils import create_presmise_index, generate_full_response, create_math_reasoning, create_reasoning
from flask import Flask, request, Response
from tools.retriever2 import Retriever
from tools.utils import create_prompt, get_model, create_math_reasoning, create_reasoning, create_presmise_index, create_guideline, generate_full_response
import json
import re
from sentence_transformers import SentenceTransformer

def parse_indices(model_output):
    # Use regex to find all numbers between <begin_idx> and <end_idx>
    pattern = r'<idx>(\d+)</idx>'
    matches = re.findall(pattern, model_output)
    # Convert matched strings to integers
    return [int(index) for index in matches]

def parser(answer):
    """
    Parse the answer string containing answer, idx, and explanation into a JSON object.
    
    Args:
        answer (str): String containing answer, idx, and explanation wrapped in their respective tags
    
    Returns:
        dict: JSON-compatible dictionary with answer, idx, and explanation
    """
    # Extract answer content
    answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
    answer_content = answer_match.group(1) if answer_match else ""
    
    # Extract idx content and convert to list of integers
    idx_match = re.search(r'<idx>(.*?)</idx>', answer, re.DOTALL)
    idx_content = idx_match.group(1) if idx_match else ""
    idx_list = [int(i) for i in idx_content.split(',') if i.strip()] if idx_content else []
    
    # Extract explanation content
    explanation_match = re.search(r'<explanation>(.*?)</explanation>', answer, re.DOTALL)
    explanation_content = explanation_match.group(1) if explanation_match else ""
    
    # Create the response dictionary
    response = {
        "answer": answer_content,
        "idx": idx_list,
        "explanation": explanation_content,
    }
    ##result = {
            ##'question': question,
            ##'answer': '',
            ##'idx': [],
            ##'explanation': '',
            ##'res': response_text,
            ##'error': '',
        ##}
    
    return response

class HmPipeline2:
    def __init__(self, embedding_model: SentenceTransformer):
        self.retriever = Retriever(embedding_model)
        self.retriever.encode("data/preprocessed_train_v1.json")
        pass

    def parse_custom_string(self, s):
        pass

    def answer(self, premises, tokenizer, model, question):
        premises_nl  = create_presmise_index(premises)

        similaries   = self.retriever.retrieve(question, threshold=0.5, top_k=3)
        print(similaries)
        flag = 1
        guide_prompt_r2 = ""
        example_prompt = ""
        if len(similaries) > 0:
            question_list = "Reference Questions:\n\n"
            for idx, x in enumerate(similaries):
                question_list += "Question Index {}:\n Premises:{}\nQuestion: {}\nAnswer: {}\nExplanation: {}\n\n".format(idx, create_presmise_index(x['premises-NL']), x['question'], x['answer'], x['explanation'])

            input_text   = question_list + "\nMain question:\n Premises:\n{}\nQuestion: {}".format(create_presmise_index(premises), question)
            #print(input_text)
            guide_prompt ="""
            You are my AI assistant. Next, I will provide you with a list of reference questions and a main question. Note that each reference question will include corresponding premises, question, answer, and explanation. Your task is to select one (or up to five) questions from the reference list that you think are most similar to the main question, meaning those questions whose information would help you answer the main question. The reference questions will be listed under the "Reference Question" section, while the main question will be under the "Main Question" section. The indices of the reference list start from 0. Ensure you use the exact Question Index as provided in the reference list for your selections. After your reasoning process, conclude your response with a list of indices of the selected reference questions, each index enclosed in <idx></idx>.
            Emphasis: You must use the exact Question Index as provided in the reference list when selecting and reporting the indices.
            """

            try:
                tt = generate_full_response(input_text, tokenizer, model, guide_prompt)
                #print(tt)
                idx_list = parse_indices(tt)

                if len(idx_list) > 0:
                    #print(idx_list)
                    example_prompt = "Preference Example:\n"
                    #print(len(similaries))
                    for idx, xx in enumerate(idx_list):
                        x = similaries[xx]
                        example_prompt += "Example Index {}:\n Premises:{}\nQuestion: {}\nAnswer: {}\nExplanation: {}\n\n".format(idx, create_presmise_index(x['premises-NL']), x['question'], x['answer'], x['explanation'])
        
                    guide_prompt_r2 = """
                    You are my AI assistant. Next, I will provide you with a list of reference questions and a main question. Note that each reference question will include corresponding premises, question, answer, and explanation. These questions and answers share similarities with the main question I will provide. Your task is to carefully and thoroughly review the reference questions before reasoning and determining how to answer the main question. Emphasis: You must meticulously analyze the reference questions, including their premises, questions, answers, and explanations, to ensure your response to the main question is well-informed and accurate.
                    """
                else:
                    flag = 0
            except:
                flag = 0
        else:
            flag = 0    

        if flag == 0:
            guide_prompt_r2 = """
            You are my AI assistant. I will provide you with premises and a question corresponding to those premises. Please think through the problem step by step and provide the answer for me.
            """
                
        #example_prompt = ""
        #print('####################################')
        input_text_r2 = "Reference Questions:\n{}\nMain Question: \nPremises: {}\nQuestion: {}".format(example_prompt, premises_nl, question)
        print(input_text_r2)
        raw_result = generate_full_response(input_text_r2, tokenizer, model, guide_prompt_r2)

        guide_prompt_r3 ="""
        You are my AI assistant. Next, I will provide you with premises, a question corresponding to those premises, and an answer along with an explanation for that question. Your task is to summarize three things for me: answer, idx, and explanation, specifically:
        answer: The answer to the question. If the answer is Yes/No, you must respond with Yes, No, or Uncertain if it cannot be determined whether the question is true or false. If the question is multiple-choice, you must output the answer (A, B, C, or D, etc.). If the question requires calculation (or "how many"), you must output a number. If the question is a mix of the above types, you must answer each part of the question as instructed above, ensuring the answers are in the same order as the questions appear. Enclose the answer in <answer></answer>.
        idx: The indices of the premises used to answer the question. Output only the numbers, separated by commas. Enclose them in <idx></idx>.
        explanation: The explanation for your answer. Read and rewrite the explanation I provided, ensuring accuracy and using only the provided information without adding any new information. Enclose it in <explanation></explanation>.
        """

        #print(generate_full_response(raw_result, tokenizer, model, guide_prompt_r3))
        result = generate_full_response(raw_result, tokenizer, model, guide_prompt_r3)
        return parser(result)

    def run(self, premises, question, tokenizer, model, trace=False):
        return self.answer(premises, tokenizer, model, question)

# if __name__ == "__main__":
#     tt = HmPipeline2()
#     premises = [
#       "Thesis eligibility requires ≥ 100 credits, GPA ≥ 5.5 (scale 0–10), capstone completion, and ≥ 80 capstone hours.",
#       "Capstone completion requires ≥ 80 credits and a 5-credit capstone course (grade ≥ 4.0).",
#       "Failed courses (grade < 4.0) add 0 credits, 0 capstone hours.",
#       "Improvement retakes (grade ≥ 4.0) use highest grade, no extra credits/hours.",
#       "Each course (grade ≥ 4.0) adds capstone hours: 3 credits = 6 hours, 4 credits = 8 hours, 5 credits = 10 hours.",
#       "Final-year students (Year 4) with capstone but < 80 hours can join capstone workshops (15 hours), if GPA ≥ 5.0.",
#       "A student (Year 3) has a GPA of 5.8, 85 credits, 100 capstone hours, no capstone course, including C1 (3 credits, 6.0, 6 hours), C2 (4 credits, 5.5, 8 hours).",
#       "The student took capstone course C3 (5 credits, 4.5), retook C1 (6.5), took C4 (3 credits, 3.8, failed), joined 2 workshops."
#     ]
#     question = "How many capstone hours has the student accumulated, and are they eligible for the thesis?"
#     tokenizer, model = get_model(model_name="Qwen/Qwen2.5-32B-Instruct-AWQ")
#     ttt = tt.run(premises, question, tokenizer, model)
#     print("##############################")
#     print(type(ttt["idx"]))
#     print(ttt['idx'])  