from prompts import guideline_prompt, which_type_prompt, math_instruction_prompt, math_prompt, reasoning_prompt, reasoning_instructions_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
import sys

def get_model(model_name = "Qwen/Qwen2.5-7B-Instruct", bnb_config = BitsAndBytesConfig(load_in_8bit=True,llm_int8_threshold=6.0,)):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model

def generate_full_response(input_text, tokenizer, model, guideline_prompt=guideline_prompt):
    messages = [
        {"role": "system", "content": guideline_prompt},
        {"role": "user", "content": input_text}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    #print(text)
    #sys.exit()
    #print("#################################################")
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids = model_inputs.input_ids
    attention_mask = (input_ids != tokenizer.eos_token_id).long()
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=10000,  
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True
    )

    # Decode toàn bộ chuỗi kết quả
    generated_sequence = outputs.sequences[0]
    full_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    
    # Chỉ lấy phần nội dung được sinh ra (bỏ phần input prompt)
    input_text_len = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    return full_text[input_text_len:].strip()

def create_prompt(instructions, question):
    return "1. Instructions:\n {}\n2. Question:\n {}".format(instructions, question)

def which_type(question, tokenizer, model):
    instructions = which_type_prompt
    text = create_prompt(instructions, question)
    return generate_full_response(text, tokenizer, model)


def create_math_reasoning(premises, question, tokenizer, model):
    reasoning = ""
    while True:
        prompt = math_prompt.format(premises, question)
        #print(prompt)
        #cur_reasoning = generate_full_response(prompt, tokenizer, model, math_instruction_prompt)
        #print("*******************************************")
        #print(cur_reasoning)
        #print("*******************************************")
        #reasoning += "\n" + cur_reasoning
        #if cur_reasoning.lower().startswith("final answer:"):
            #return reasoning
        return generate_full_response(prompt, tokenizer, model, math_instruction_prompt)

def create_reasoning(premises, question, reasoning, tokenizer, model):
    prompt = reasoning_prompt.format(premises, question, reasoning)
    return generate_full_response(prompt, tokenizer, model, reasoning_instructions_prompt)

def create_presmise_index(premises):
    preprocessed_premises = ""
    for idx, pm in enumerate(premises):
        preprocessed_premises += "Premise {}: {}\n".format(idx + 1, pm)
    return preprocessed_premises


def create_guideline(question, premises_nl, premises_fol, retriever, tokenizer, model):
    results = retriever.retrieve(question)
    if len(results) == 0:
        return "There isn't any guideline!"
    
    input_text = ""
    for idx, re in enumerate(results):
        input_text += "Example {}:\n".format(idx + 1)
        
        input_text += "Premises in form of Natural Language:\n{}\n".format(create_presmise_index(re['premises-NL']))

        input_text += "Premises in form of First Order Logic:\n{}\n".format(create_presmise_index(re['premises-FOL']))

        input_text += "Question: \n{}\n".format(re['question'])

        input_text += "Answer for the above question: \n{}\n".format(re['answer'])

        input_text += "Explanation for the above answer: \n{}\n".format(re['explanation'])

    input_text += "That's all examples! Below is my premises-NL, premises-FOL and question:\n"
    input_text += "Premises-NL: \n{}\n".format(premises_nl)
    input_text += "Premises-FOL: \n{}\n".format(premises_fol)
    input_text += "Question: \n{}\n".format(question)
    input_text += "Now, it's your turn to response"

    #guideline = """
        #You are my assistant. I will provide you with a series of examples (each labeled with a number, note that it's possible to have no examples). In each example, the following elements are included:

        #1. Premises in Natural Language (Premises-NL): These are statements expressed in natural language.
        #2. Premises in First Order Logic (Premises-FOL): These are the same statements expressed using first-order logic.
        #3. Question: A question related to the above premises.
        #4. Answer: The correct answer to the above question.
        #5. Explanation: A step-by-step reasoning process explaining how the answer was derived, based on both the natural language and logical representations.

        #After you've reviewed the examples, I will give you a new set of Premises-NL, Premises-FOL, and a Question.

        #Your task is not to solve the question.
        #Instead, write a clear, step-by-step guideline or reasoning framework that outlines how one should approach answering the question.
        #The guideline should:

        #1. Be based on the reasoning patterns shown in the examples,
        #2. Be logically coherent and well-structured,
        #3. Not include the actual answer or specific steps that lead directly to it.

        #Think of it as writing a template or problem-solving strategy for similar types of questions.
    #"""

    guideline="""
You are my assistant. I will provide you with a series of examples (labeled as Example 1, Example 2, etc.), where each example includes:

1. Premises in Natural Language (Premises-NL): Statements expressed in natural language.
2. Premises in First-Order Logic (Premises-FOL): Statements expressed in first-order logic.
3. Question: A question related to the premises.
4. Answer: The answer to the question.
5. Explanation: A detailed explanation of why the answer is correct, including the reasoning process.

Your task is to analyze the examples to understand the reasoning process used to answer the questions. Then, I will provide new Premises-NL, Premises-FOL, and a Question. Do not solve the question or provide the answer or explanation for it. Instead, based on the reasoning patterns in the examples, write a detailed, clear, coherent, and accurate step-by-step guide explaining how to solve the question. The guide should:

1. Describe the general process for solving the problem, tailored to the structure of my premises and question.
2. Include specific steps (e.g., how to use the Premises-FOL, apply logical rules, or interpret the question).
3. Be informed by the reasoning methods shown in the examples (e.g., inference rules, predicate logic manipulation).
4. Avoid giving the final answer, partial solutions, or any explanation specific to the outcome of my question.
5. Ensure the guide is precise, easy to follow, and applicable to similar problems while being customized to the format of my premises and question.
    """

    return generate_full_response(input_text, tokenizer, model, guideline)
