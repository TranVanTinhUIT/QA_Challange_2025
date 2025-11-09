from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flasgger import Swagger
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from datetime import timedelta, datetime
import os
import json
from sentence_transformers import SentenceTransformer
import time

from similar_classification import SimilarClassification
from choice_pipeline_4 import ChoicePipeline4
from yes_no_pipeline_2 import YesNoPipeline2
# from hm_pipeline2 import HmPipeline2  
from hm_pipeline3 import HmPipeline3  

# Khởi tạo Flask app
app = Flask(__name__)
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "My API",
        "description": "API with Bearer Token auth",
        "version": "1.0"
    },
    "securityDefinitions": {
        "Bearer": {
            "type": "apiKey",
            "name": "Authorization",
            "in": "header",
            "description": "Enter: **Bearer &lt;JWT token&gt;**"
        }
    },
    "security": [
        {
            "Bearer": []
        }
    ],
    "host": "sktt1-xai.work:443",
    "schemes": ["http"]
}
swagger = Swagger(app, template=swagger_template)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["10 per second", "100 per hour"]
)

app.config["JWT_SECRET_KEY"] = "3f9e4302dd42404485369b13d2988ff0cae93f1efbcc47ddb7d0c38ed23bcb83a0083d640e794622a5c47b206bbbc2329b9c931969a040d5935977e6493606d3"
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=30)
jwt = JWTManager(app)

def get_model(model_name = "Qwen/Qwen2.5-7B-Instruct", bnb_config = BitsAndBytesConfig(load_in_8bit=True,llm_int8_threshold=6.0,)):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model

# tokenizer, model = get_model(model_name="Qwen/Qwen3-32B-AWQ") # Model
# tokenizer, model = get_model(model_name="Qwen/Qwen2.5-7B-Instruct") # Model
# tokenizer, model = get_model(model_name="Qwen/Qwen2.5-32B-Instruct-AWQ") # Model
tokenizer, model = get_model() # Model

embedding_model = SentenceTransformer('BAAI/bge-m3')

yn_index_path= "/home/tinhtv/code/RAG/api/datastore/yn_index"
choice_index_path= "/home/tinhtv/code/RAG/api/datastore/choice_index"
hm_index_path= "/home/tinhtv/code/RAG/api/datastore/hm_index"
q_classifier = SimilarClassification(yn_index_path=yn_index_path, choice_index_path=choice_index_path, hm_index_path=hm_index_path) # Instance of question classifier

choice_pipeline = ChoicePipeline4()
yesno_pipeline = YesNoPipeline2()
# hm_pipeline = HmPipeline2(embedding_model=embedding_model)  
hm_pipeline = HmPipeline3(embedding_model=embedding_model)  

def validate(request):
    """
        Return error message or None
    """

    # Body is JSON format
    if not request.is_json:
        return 'Unexpected, require body is JSON format'

    data = request.get_json()

    # `premises` field
    if 'premises-NL' not in data:
        return 'Require `premises-NL` field'

    premises = data.get('premises-NL')

    if not isinstance(premises, list) :
        return 'Unexpected, require `premises-NL` is an array.'
    
    if len(premises) == 0:
        return 'Unexpected, premises-NL is empty'
    
    if not all(isinstance(premise, str) for premise in premises):
        return 'Unexpected, require premises-NL is an array of string'

    # `question` field
    if 'question' not in data:
        return 'Require `question` field'
    
    question = data.get('question')

    if not isinstance(question, str):
        return 'Unexpected, require `question` is a string.'
    
    return None

USER_DICT = {
    'admin': 'adminpw@123',
    'Tinh' : 'Tinhpw@123',
    'Manh' : 'Manhpw@123',
    'Hoan' : 'Hoanpw@123',
    'Quan' : 'Quanpw@123',
    'Khoa' : 'Khoapw@123',
    'Khang': 'Khangpw@123'
}

temp_records = [] 
temp_records_err = []

def check_credentials(username, password):
    if username in USER_DICT and USER_DICT[username] == password:
        return True
    return False

@app.route("/login", methods=["POST"])
@limiter.limit('3 per minute')
def login():
    """
    Login
    ---
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - username
            - password
          properties:
            username:
              type: string
            password:
              type: string
    responses:
      200:
        description: Response
        examples:
          application/json: { "token": "abc..."  }
    """
    data = request.get_json()
    username = data.get("username", '')
    password = data.get("password", '')
    if not check_credentials(username=username, password=password):
        return jsonify({"msg": "Bad username or password"}), 401

    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

# Tùy chỉnh response 401 khi không có token
@jwt.unauthorized_loader
def custom_unauthorized_response(error):
    return jsonify({
        'error': 'Unauthorized',
        'message': 'Missing or invalid token'
    }), 401

# Tùy chỉnh response 401 khi token không hợp lệ
@jwt.invalid_token_loader
def custom_invalid_token_response(error):
    return jsonify({
        'error': 'Unauthorized',
        'message': 'Token is invalid'
    }), 401

# Endpoint nhận POST và trả về kết quả hoàn chỉnh
@app.route("/query", methods=["POST"])
@limiter.limit('10 per second')
@jwt_required()
def generate():
    """
    query
    ---
    security:
      - Bearer: []
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - premises-NL
            - question
          properties:
            premises-NL:
              type: array
              items:
                type: string
            question:
              type: string
    responses:
      200:
        description: Response
        examples:
          application/json: { "answer": "Yes", "idx": [1, 2, 3], "explanation": "From Premise 3, we know...."  }
    """
    # Check json format
    error_message = validate(request)
    if error_message:
        error = {
            'message': error_message
        }
        return jsonify(error), 400

    data = request.get_json()

    try:
        
        premises = data.get("premises-NL", [])

        question = data.get("question", '')

        action = 'classify'
        # Clasify question
        question_type, scores = q_classifier.classify(question, embedding_model=embedding_model, k=10, trace=False)
        print(f'[Classification] ')
        print(f'Question: {question}  ')
        print(f'Question types => {question_type}  ')

        # Answer the question
        action = 'answer'
        response_json = {
            'answers': '',
            'idx': [],
            'explanation': ''
        }

        start_time = time.perf_counter()
        if question_type == 1: # Yes/No/Uncertain
            yn_result = yesno_pipeline.run(premises=premises, question=question, tokenizer=tokenizer, model=model, trace= True)
            # yn_result = yn_results[0]  # Get the first (and only) result from the list
            response_json['answers'] = yn_result.get('answer', 'Error')
            response_json['idx'] = yn_result.get('idx', [])
            response_json['explanation'] = yn_result.get('explanation', 'Error processing request.')
        elif question_type == 2: # Multiple choice
            choice_answer  = choice_pipeline.run(premises=premises, question=question, tokenizer=tokenizer, model=model, trace=True)
            response_json['answers'] = choice_answer['answer']
            response_json['idx'] = choice_answer['idx']
            response_json['explanation'] = choice_answer['explanation']
        else:  # How many
            # Old pipeline code
            # hm_result = hm_pipeline.run(premises, question, tokenizer, model, trace= True)
            # response_json['answers'] = hm_result['answer']
            # response_json['idx'] = hm_result['idx']
            # response_json['explanation'] = hm_result['explanation']
            
            # New pipeline code
            hm_result = hm_pipeline.run(premises, question, tokenizer, model, trace= True)
            response_json['answers'] = hm_result['answer']
            response_json['idx'] = hm_result['idx']
            response_json['explanation'] = hm_result['explanation']

        end_time = time.perf_counter()
        # Store request and response
        temp_records.append({
            'moment': datetime.now().isoformat(),
            'time': end_time- start_time,
            'q_types': question_type,
            'req': data,
            'res': response_json
        })

        return jsonify(response_json), 200

    except Exception as e:
        full_stack_trace = traceback.format_exc()
        temp_records_err.append({
            'time': datetime.now().isoformat(),
            'req': data,
            'err':  f'Error: {str(e)}, {full_stack_trace}'
        })

        error = {
            'message': f"Action: {action}, Error: {str(e)}, {full_stack_trace}"
        }
        return jsonify(error), 500

@app.route("/save-temp", methods=["POST"])
@jwt_required()
def save_temp():
    # TODO: save temp records
    nowstr = datetime.now().isoformat()
    success_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive", f"success_{nowstr}.json")
    with open(success_path, 'w', encoding='utf-8') as f:
        json.dump(temp_records, f, indent=2, ensure_ascii=False)

    error_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive", f"error_{nowstr}.json")
    with open(error_path, 'w', encoding='utf-8') as f:
        json.dump(temp_records_err, f, indent=2, ensure_ascii=False)
    
    print('[Save temp records] '+ nowstr)
    return '', 200

# Chạy server Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
