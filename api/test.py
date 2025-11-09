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
    def __init__(self, embedding_model):
        self.retriever = Retriever(embedding_model=embedding_model)
        self.retriever.encode("data/preprocessed_train_v1.json")
        pass

    def parse_custom_string(self, s):
        pass

    def answer(self, premises, tokenizer, model, question):
        premises_nl  = create_presmise_index(premises)

        similaries   = self.retriever.retrieve(question, threshold=0.5, top_k=3)
        flag = 1
        guide_prompt_r2 = ""
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

if __name__ == "__main__":
    embedding_model = SentenceTransformer('BAAI/bge-m3')
    tt = HmPipeline2(embedding_model=embedding_model)
    premises= [
     "Nam has a GPA of 6.9 after completing 88 credits.",
      "Nam retook 3 previously passed courses: C1 (4 credits, initial 5.2, retake1 7.0, retake2 3.5), C2 (3 credits, initial 6.0, retake1 6.8, retake2 6.0), C3 (5 credits, initial 7.5, retake1 5.0, retake2 4.0).",
      "Nam repeated one failed course: C4 (3 credits, initial 3.0, repeat 6.5).",
      "Nam took two new courses: C5 (4 credits, score 8.0), C6 (2 credits, score 3.8).",
      "A retake is re-registering a passed course (same course code) to improve the grade.",
      "For GPA, use the highest score if >= 4.0; if all retake scores < 5.0, deduct 0.2 points per credit from the course's grade points.",
      "Retaking a course does not add credits.",
      "Repeating a failed course means re-registering it.",
      "A failed course (score < 4.0) adds no credits.",
      "Repeating a failed course adds credits if the new score >= 4.0."
    ]
    question = "What is Nam's updated GPA after all course attempts?"
    tokenizer, model = get_model(model_name="Qwen/Qwen2.5-32B-Instruct-AWQ")
    ttt = tt.run(premises, question, tokenizer, model)
    print(ttt)
    print("##############################")
    print(type(ttt["idx"]))
    print(ttt['idx'])  