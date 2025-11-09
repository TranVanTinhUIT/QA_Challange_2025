from flask import Flask, request, Response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tools.prompts import bi_prompt, mul_prompt, hm_prompt
from tools.utils import create_prompt, get_model, create_math_reasoning, create_reasoning

# Khởi tạo Flask app
app = Flask(__name__)

tokenizer, model           = get_model(model_name="Qwen/Qwen2.5-Math-7B-Instruct")
#tokenizer_math, model_math = get_model(model_name="Qwen/Qwen2.5-Math-7B-Instruct")
tokenizer_mat, model_math = tokenizer, model

from tools.utils import generate_full_response, which_type

# Endpoint nhận POST và trả về kết quả hoàn chỉnh
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        premises_nl = data.get("premises-NL", [])
        questions   = data.get("questions", [])
        #print(questions)

        premises = ""
        for idx, pm in enumerate(premises_nl):
            premises += "Premise {}: {}\n".format(idx + 1, pm)

        response_text = ""
        for question in questions:
            math_reasoning = create_math_reasoning(premises, question, tokenizer, model)
            print("=============================================================")
            print(math_reasoning)
            print("=============================================================")

            print("#############################################################")
            response_text  += "\n" + create_reasoning(premises, question, math_reasoning, tokenizer, model)
            print("#############################################################")
            print()

        return Response(response_text, mimetype="text/plain")

    except Exception as e:
        return Response(f"Error: {str(e)}", status=500, mimetype="text/plain")

# Chạy server Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
