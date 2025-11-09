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

def get_question_type(question):
    """
    Determine the type of question to apply specific formatting rules
    """
    question = question.lower()
    if "fee" in question or "cost" in question or "price" in question:
        return "monetary"
    elif "and" in question or "," in question:
        return "chained"
    elif "how many" in question:
        return "numerical"
    elif "yes" in question or "no" in question:
        return "boolean"
    return "general"

def format_answer(answer, question_type):
    """
    Format the answer based on question type
    """
    if question_type == "monetary":
        return f"${answer.strip('$')}"
    elif question_type == "chained":
        parts = [part.strip() for part in answer.split(',')]
        return ", ".join(parts)
    elif question_type == "numerical":
        return answer.strip()
    elif question_type == "boolean":
        return answer.strip().capitalize()
    return answer.strip()

def parser(answer):
    """
    Parse the answer string containing answer, idx, and explanation into a JSON object.
    """
    try:
        # Extract answer content and clean it
        answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
        answer_content = answer_match.group(1) if answer_match else ""
        
        # Clean answer content
        answer_content = answer_content.strip()
        # Remove newlines and extra spaces
        answer_content = re.sub(r'\n+', ' ', answer_content)
        answer_content = re.sub(r'\s+', ' ', answer_content)
        
        # Handle special cases
        if answer_content.startswith('$'):
            # Keep the dollar sign
            answer_content = answer_content.strip()
        elif ',' in answer_content:
            # Handle chained answers (e.g., "28, No")
            parts = [part.strip() for part in answer_content.split(',')]
            answer_content = ', '.join(parts)
        
        # Extract idx content and convert to list of integers
        idx_match = re.search(r'<idx>(.*?)</idx>', answer, re.DOTALL)
        idx_content = idx_match.group(1) if idx_match else ""
        idx_list = [int(i) for i in idx_content.split(',') if i.strip()] if idx_content else []
        
        # Extract explanation content and clean it
        explanation_match = re.search(r'<explanation>(.*?)</explanation>', answer, re.DOTALL)
        explanation_content = explanation_match.group(1) if explanation_match else ""
        explanation_content = explanation_content.strip()
        
        # If no proper answer was found, try to extract from the raw text
        if not answer_content or answer_content == "Error: No answer generated":
            # Try to find a number in the text
            number_match = re.search(r'\$?\d+(?:\.\d+)?', answer)
            if number_match:
                answer_content = number_match.group(0)
            else:
                # Try to find Yes/No/Uncertain
                yn_match = re.search(r'\b(Yes|No|Uncertain)\b', answer, re.IGNORECASE)
                if yn_match:
                    answer_content = yn_match.group(1)
        
        # If no proper explanation was found, use the raw text
        if not explanation_content or explanation_content == "Error: No explanation generated":
            explanation_content = answer.strip()
        
        # If no indices were found, try to extract from the text
        if not idx_list:
            # Look for premise references in the text
            idx_pattern = r'Premise (\d+)'
            idx_matches = re.findall(idx_pattern, answer)
            if idx_matches:
                idx_list = [int(idx) for idx in idx_matches]
            else:
                # If still no indices, use all premises
                idx_list = list(range(1, len(premises) + 1))
        
        return {
            "answer": answer_content,
            "idx": idx_list,
            "explanation": explanation_content,
        }
    except Exception as e:
        print(f"Error in parser: {str(e)}")
        # Try to extract any useful information from the raw answer
        try:
            # Look for any number
            number_match = re.search(r'\$?\d+(?:\.\d+)?', answer)
            if number_match:
                return {
                    "answer": number_match.group(0),
                    "idx": list(range(1, len(premises) + 1)),
                    "explanation": answer.strip()
                }
        except:
            pass
        
        return {
            "answer": "Error: Failed to parse answer",
            "idx": list(range(1, len(premises) + 1)),
            "explanation": f"Error: {str(e)}"
        }

def validate_answer(answer_item, question):
    """
    Validate the answer format and content based on question type
    """
    answer = answer_item["answer"]
    idx = answer_item["idx"]
    explanation = answer_item["explanation"]
    
    # Check for empty answer
    if not answer or answer.isspace():
        return False, "Empty answer"
        
    # Check for empty explanation
    if not explanation or explanation.isspace():
        return False, "Empty explanation"
        
    # Check for empty idx when answer is not "Uncertain"
    if not idx and "Uncertain" not in answer:
        return False, "Missing premise indices"
        
    # Check for monetary values
    if "fee" in question.lower() or "cost" in question.lower() or "price" in question.lower():
        if not answer.startswith("$"):
            return False, "Missing currency symbol for monetary value"
            
    # Check for chained answers
    if "and" in question.lower() or "," in question:
        if "," not in answer:
            return False, "Missing comma in chained answer"
            
    # Check for numerical answers
    if "how many" in question.lower():
        if not any(c.isdigit() for c in answer):
            return False, "Missing numerical value"
            
    return True, "Valid answer"

class HmPipeline3:
    def __init__(self, embedding_model: SentenceTransformer):
        self.retriever = Retriever(embedding_model)
        self.retriever.encode("data/preprocessed_train_v1.json")
        pass

    def parse_custom_string(self, s):
        pass

    def answer(self, premises, tokenizer, model, question):
        try:
            premises_nl = create_presmise_index(premises)

            similaries = self.retriever.retrieve(question, threshold=0.5, top_k=3)
            print(similaries)
            flag = 1
            guide_prompt_r2 = ""
            example_prompt = ""
            if len(similaries) > 0:
                question_list = "Reference Questions:\n\n"
                for idx, x in enumerate(similaries):
                    question_list += "Question Index {}:\n Premises:{}\nQuestion: {}\nAnswer: {}\nExplanation: {}\n\n".format(idx, create_presmise_index(x['premises-NL']), x['question'], x['answer'], x['explanation'])

                input_text = question_list + "\nMain question:\n Premises:\n{}\nQuestion: {}".format(create_presmise_index(premises), question)
                guide_prompt ="""
                You are my AI assistant. Next, I will provide you with a list of reference questions and a main question. Note that each reference question will include corresponding premises, question, answer, and explanation. Your task is to select one (or up to five) questions from the reference list that you think are most similar to the main question, meaning those questions whose information would help you answer the main question. The reference questions will be listed under the "Reference Question" section, while the main question will be under the "Main Question" section. The indices of the reference list start from 0. Ensure you use the exact Question Index as provided in the reference list for your selections. After your reasoning process, conclude your response with a list of indices of the selected reference questions, each index enclosed in <idx></idx>.
                Emphasis: You must use the exact Question Index as provided in the reference list when selecting and reporting the indices.
                """

                try:
                    tt = generate_full_response(input_text, tokenizer, model, guide_prompt)
                    idx_list = parse_indices(tt)

                    if len(idx_list) > 0:
                        example_prompt = "Preference Example:\n"
                        for idx, xx in enumerate(idx_list):
                            x = similaries[xx]
                            example_prompt += "Example Index {}:\n Premises:{}\nQuestion: {}\nAnswer: {}\nExplanation: {}\n\n".format(idx, create_presmise_index(x['premises-NL']), x['question'], x['answer'], x['explanation'])
            
                        guide_prompt_r2 = """
                        You are my AI assistant. Next, I will provide you with a list of reference questions and a main question. Note that each reference question will include corresponding premises, question, answer, and explanation. These questions and answers share similarities with the main question I will provide. Your task is to carefully and thoroughly review the reference questions before reasoning and determining how to answer the main question. Emphasis: You must meticulously analyze the reference questions, including their premises, questions, answers, and explanations, to ensure your response to the main question is well-informed and accurate.
                        """
                    else:
                        flag = 0
                except Exception as e:
                    print(f"Error in reference question processing: {str(e)}")
                    flag = 0
            else:
                flag = 0    

            if flag == 0:
                guide_prompt_r2 = """
                You are my AI assistant. I will provide you with premises and a question corresponding to those premises. Please think through the problem step by step and provide the answer for me. Remember to:
                1. Clearly state your answer
                2. List the premises you used
                3. Provide a detailed explanation
                """

            input_text_r2 = "Reference Questions:\n{}\nMain Question: \nPremises: {}\nQuestion: {}".format(example_prompt, premises_nl, question)
            print(input_text_r2)
            raw_result = generate_full_response(input_text_r2, tokenizer, model, guide_prompt_r2)

            guide_prompt_r3 ="""
            You are my AI assistant. Next, I will provide you with premises, a question corresponding to those premises, and an answer along with an explanation for that question. Your task is to summarize three things for me: answer, idx, and explanation, specifically:

            1. answer: The answer to the question. Follow these rules:
               - For monetary values: Include $ symbol (e.g., "$100" not "100")
               - For chained answers (multiple parts): Use "Number, Yes/No" format (e.g., "28, No")
               - For numerical answers: 
                 * Include ONLY the final calculated number
                 * For GPA, use 2 decimal places (e.g., "5.86")
                 * For percentages, include % symbol
                 * Do not include any units or descriptive text
               - For yes/no questions: Use ONLY "Yes" or "No"
               - For calculations:
                 * Show ONLY the final result
                 * Do not include intermediate steps
                 * Round to appropriate decimal places
                 * For GPA calculations, always show 2 decimal places
               - Do not add newlines or extra spaces
               - Keep the answer concise and in the exact format required
               - IMPORTANT: The answer must be the final calculated value, not a placeholder or intermediate step
               Enclose the answer in <answer></answer>

            2. idx: The indices of the premises used to answer the question. Rules:
               - Include ALL premises that were used in the reasoning
               - Even if the answer seems obvious, include the premises that support it
               - For calculations, include premises with the formulas and rules used
               - For chained questions, include premises for each part of the answer
               - Numbers should be in ascending order
               Output only the numbers, separated by commas. Enclose them in <idx></idx>

            3. explanation: The explanation for your answer. Rules:
               - Be clear and concise
               - Focus on the key steps that led to the answer
               - For calculations:
                 * Show only the essential calculation steps
                 * Include the final formula used
                 * Show the final result
                 * For GPA calculations, show:
                   - Total credits
                   - Total grade points
                   - Final GPA calculation
               - For chained questions:
                 * Explain each part separately
                 * Show how each part was determined
               - Reference the premises used in each step
               - Do not include unnecessary details or intermediate steps
               - Keep the explanation focused on how you arrived at the final answer
               Enclose it in <explanation></explanation>

            IMPORTANT REMINDERS:
            1. The answer must be the final calculated value
            2. For GPA calculations, always show 2 decimal places
            3. Do not include any intermediate steps in the answer
            4. The explanation should be concise but show the key calculation steps
            5. Make sure to reference the premises used in your calculations
            """

            # Get result from prompt3
            result = generate_full_response(raw_result, tokenizer, model, guide_prompt_r3)
            parsed_result = parser(result)

            # Add prompt4 to convert to JSON format
            guide_prompt_r4 = """
            You are my AI assistant. I will provide you with a previous answer that needs to be converted into a specific JSON format. The input will contain:
            1. The answer to the question
            2. The premises used (indices)
            3. The explanation

            Your task is to convert this information into a JSON object with the following structure:
            {
                "answer": "the answer string",
                "explanation": "the explanation string",
                "premises_used": [list of premise numbers]
            }

            Rules for the answer field:
            1. For monetary values: Include $ symbol (e.g., "$100")
            2. For chained answers: Use "Number, Yes/No" format (e.g., "28, No")
            3. For numerical answers: Include only the number
            4. For yes/no questions: Use only "Yes" or "No"
            5. Do not include any additional text or explanations
            6. Keep the answer concise and in the exact format required

            Rules for the explanation field:
            1. Keep the explanation clear and concise
            2. Include all necessary context
            3. Reference the premises used
            4. Show calculations if applicable

            Rules for premises_used:
            1. Must be a list of numbers
            2. Numbers should be in ascending order
            3. Include all premises that contributed to the answer

            Please ensure your response is valid JSON and follows this structure exactly.
            """

            # Create input for prompt4 using the parsed result from prompt3
            prompt4_input = f"""
            Previous answer: {parsed_result['answer']}
            Previous explanation: {parsed_result['explanation']}
            Previous premises used: {parsed_result['idx']}
            """

            # Get JSON response from prompt4
            json_result = generate_full_response(prompt4_input, tokenizer, model, guide_prompt_r4)
            
            try:
                # Parse the JSON response
                json_data = json.loads(json_result)
                
                # Create the final result
                final_result = {
                    "answer": json_data.get("answer", ""),
                    "idx": json_data.get("premises_used", []),
                    "explanation": json_data.get("explanation", "")
                }
                
                # Validate the result
                is_valid, validation_message = validate_answer(final_result, question)
                
                if not is_valid:
                    print(f"Validation failed: {validation_message}")
                    # If invalid, try to fix common issues
                    if "Missing currency symbol" in validation_message:
                        final_result["answer"] = f"${final_result['answer'].strip('$')}"
                    elif "Missing comma" in validation_message:
                        # Try to split and rejoin with proper formatting
                        parts = [part.strip() for part in final_result["answer"].split()]
                        final_result["answer"] = ", ".join(parts)
                        
                    # If still invalid, regenerate with more specific prompt
                    if not validate_answer(final_result, question)[0]:
                        specific_prompt = f"""
                        The previous answer was invalid: {validation_message}
                        Please provide a new answer following these specific rules:
                        {validation_message}
                        """
                        result = generate_full_response(raw_result, tokenizer, model, specific_prompt)
                        final_result = parser(result)
                
                # Format answer based on question type
                question_type = get_question_type(question)
                final_result["answer"] = format_answer(final_result["answer"], question_type)
                
                # Final validation before returning
                if not final_result["answer"] or final_result["answer"] == "Error: No answer generated":
                    # Try to extract answer from raw result
                    number_match = re.search(r'\$?\d+(?:\.\d+)?', raw_result)
                    if number_match:
                        final_result["answer"] = number_match.group(0)
                    else:
                        final_result["answer"] = raw_result.strip()
                
                if not final_result["explanation"] or final_result["explanation"] == "Error: No explanation generated":
                    final_result["explanation"] = raw_result.strip()
                
                if not final_result["idx"]:
                    final_result["idx"] = list(range(1, len(premises) + 1))
                
                return final_result
                
            except json.JSONDecodeError:
                print("Failed to parse JSON response, falling back to parsed result from prompt3")
                return parsed_result
            
        except Exception as e:
            print(f"Error in answer function: {str(e)}")
            # Try to extract any useful information
            try:
                number_match = re.search(r'\$?\d+(?:\.\d+)?', raw_result)
                if number_match:
                    return {
                        "answer": number_match.group(0),
                        "idx": list(range(1, len(premises) + 1)),
                        "explanation": raw_result.strip()
                    }
            except:
                pass
            
            return {
                "answer": raw_result.strip() if raw_result else "Error: Processing failed",
                "idx": list(range(1, len(premises) + 1)),
                "explanation": f"Error occurred during processing: {str(e)}"
            }

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