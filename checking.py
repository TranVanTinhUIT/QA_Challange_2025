from tools.retriever import Retriever
from flask import Flask, request, Response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tools.prompts import bi_prompt, mul_prompt, hm_prompt
from tools.utils import create_prompt, get_model, create_math_reasoning, create_reasoning, create_presmise_index, generate_full_response
import sys

# Khởi tạo Flask app
app = Flask(__name__)

tokenizer, model = get_model(model_name="Qwen/Qwen2.5-14B-Instruct")

# Endpoint nhận POST và trả về kết quả hoàn chỉnh
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        premises_nl  = create_presmise_index(data.get("premises-NL", []))
        premises_fol = create_presmise_index(data.get("premises-FOL", []))
        question     = data.get("question", "")
        answer       = data.get("answer", "")
        explanation  = data.get("explanation", "")
        idx          = data.get("idx", [])

        guideline="""
You are my assistant, responsible for checking whether a dataset has any issues with logic or format. I will provide you with each data point, structured as follows:  
- Premises-NL: Statements in natural language.  
- Premises-FOL: Statements in first-order logic.  
- Question: The question.  
- Answer: The answer.  
- Explanation: Explanation for the provided answer.  
- idx: The premises used in the explanation.  

I need you to verify whether the provided data point is valid by checking the following:  
- The premises listed in idx must be consistent with the explanation.  
- The explanation must be consistent with the answer  
- Premises-NL must be consistent with Premises-FOL  
- The premises used in the explanation must be consistent with the premises provided in Premises-NL and Premises-FOL.  
- All calculations, if any, must be performed accurately.  

Additionally, there may be other logical errors I haven not considered. Your task is to thoroughly review and identify any issues.  

Please provide the output in the following format:  
{
  "answer": "",
  "explanation": ""
}
Where answer is "OK" if the data point has no issues, or "Have problems" if issues are found. The explanation field should explain the issues if answer is "Have problems"; otherwise, leave it empty if answer is "OK".
"""

        input_text = "Below is my data point:\n"
        input_text += "Premise-NL: \n{}\n".format(premises_nl)
        input_text += "Premise-FOL: \n{}\n".format(premises_fol)
        input_text += "Question: \n{}\n".format(question)
        input_text += "Answer: \n{}\n".format(answer)
        input_text += "Explanation: \n{}\n".format(explanation)
        input_text += "idx: \n{}\n".format(idx)
        input_text += "Now, it is your turn to response"

        response_text = generate_full_response(input_text, tokenizer, model, guideline)
        
        return Response(response_text, mimetype="text/plain")

    except Exception as e:
        return Response(f"Error: {str(e)}", status=500, mimetype="text/plain")

# Chạy server Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
