import re
import json
import torch
import logging
from tools.utils import create_presmise_index, generate_full_response, create_math_reasoning, create_reasoning

class HmPipeline:
    def __init__(self):
        pass

    def parse_custom_string(self, s):
        # Initialize result dictionary
        result = {}
    
        # Remove curly braces and clean up string
        s = s.strip().strip('{}').replace('\n', '').strip()
    
        # Find and extract "answer"
        answer_start = s.find('"answer":')
        if answer_start != -1:
            answer_value_start = s.find('"', answer_start + 9) + 1
            answer_value_end = s.find('"', answer_value_start)
            result['answer'] = s[answer_value_start:answer_value_end]
    
        # Find and extract "idx"
        idx_start = s.find('"idx":')
        if idx_start != -1:
            idx_value_start = s.find('[', idx_start)
            idx_value_end = s.find(']', idx_value_start) + 1
            idx_str = s[idx_value_start:idx_value_end]
            # Convert idx string to list of integers
            numbers = idx_str.strip('[]').split(',')
            result['idx'] = [int(num.strip()) for num in numbers if num.strip()]
    
        # Find and extract "explanation"
        explanation_start = s.find('"explanation":')
        if explanation_start != -1:
            explanation_value_start = s.find('"', explanation_start + 13) + 1
            explanation_value_end = s.rfind('"')
            result['explanation'] = s[explanation_value_start:explanation_value_end]
    
        return result

    def run(self, premises, question, tokenizer, model, trace=False):
        premises = create_presmise_index(premises)
        with torch.no_grad():
            math_reasoning = create_math_reasoning(premises, question, tokenizer, model)
            response_text  = create_reasoning(premises, question, math_reasoning, tokenizer, model)

        result = {
            'question': question,
            'answer': '',
            'idx': [],
            'explanation': '',
            'res': response_text,
            'error': '',
        }
        tt = self.parse_custom_string(response_text)
        print(response_text)
        print(tt)
        print()
        result['idx'] = tt['idx']
        result['answer'] = tt['answer']
        result['explanation'] = tt['explanation']
        return result
    


from flask import Flask, request, Response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tools.prompts import bi_prompt, mul_prompt, hm_prompt
from tools.utils import create_prompt, get_model, create_math_reasoning, create_reasoning

# Khởi tạo Flask app
app = Flask(__name__)

tokenizer, model           = get_model(model_name="Qwen/Qwen2.5-7B-Instruct")
#tokenizer_math, model_math = get_model(model_name="Qwen/Qwen2.5-Math-7B-Instruct")
tokenizer_mat, model_math = tokenizer, model

from tools.utils import generate_full_response, which_type

hm = HmPipeline()

# Endpoint nhận POST và trả về kết quả hoàn chỉnh
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        premises_nl = data.get("premises-NL", [])
        question    = data.get("question", "")
        #print(questions)

        result = hm.run(premises_nl, question, tokenizer, model) 

        return Response(result['res'], mimetype="text/plain")

    except Exception as e:
        return Response(f"Error: {str(e)}", status=500, mimetype="text/plain")

# Chạy server Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
