import re
import json
import torch
import logging

class HmPipeline:
    def __init__(self):
        pass

    def generate_prompt(self, premises, question):
        premise_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises)])
        return f"""
            Solve this question using Graph of Thought with two concise reasoning paths, provide a very concise answer.

            Premises:
            {premise_text}

            Question:
            {question}

            For each path:
            1. PATH A (1-2 sentences): Use premises [specify numbers] to solve directly
            2. PATH B (1-2 sentences): Use a different approach with premises [specify different numbers if possible]

            You do not need to use all premises. You can use any combination of them.
            Each path must be under 50 words. State only the relevant premise numbers and your conclusion in each path.

            Based on your two paths, generate a final most correct and logical answer to the question. 

            Return the answer in this JSON format:
            ```json
            {{
            "answer": "<your answer by words or numbers>",
            "idx": <relevant premise numbers as integers>,
            "explanation": "<your explanation in 50 words or fewer how you got the answer using the premises.">" 
            }}
            """

    def parse_response(self, response):
        
        # Try to extract JSON from code blocks first
        matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if matches:
            try:
                parsed = json.loads(matches[0])
                return parsed
            except json.JSONDecodeError as e:
                # Fallback: Extract fields using regex
                answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', response)
                idx_match = re.findall(r'"idx"\s*:\s*\[([^\]]*)\]', response)
                explanation_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', response)
                if answer_match and idx_match and explanation_match:
                    answer = answer_match.group(1)
                    idx_str = idx_match[0]
                    try:
                        idx = [int(i.strip()) for i in idx_str.split(',') if i.strip().isdigit()]
                    except:
                        idx = []
                    explanation = explanation_match.group(1)

                    return {
                        "answer": answer,
                        "idx": idx,
                        "explanation": explanation
                    }
        # If no code blocks or regex extraction failed, try broader patterns

        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    return parsed
                except json.JSONDecodeError:
                    pass
        except:
            pass
        # Last attempt: extract fields directly from text
        answer_match = re.search(r'answer["\s:]+([^"\n]+)', response, re.IGNORECASE)
        idx_match = re.findall(r'(?:premise|idx|indices)["\s:]+(?:\[)?([0-9\s,]+)(?:\])?', response, re.IGNORECASE)
        explanation_match = re.search(r'explanation["\s:]+([^"\n]+)', response, re.IGNORECASE)
        if answer_match:
            result = {"answer": answer_match.group(1).strip()}
            if idx_match:
                try:
                    idx_str = idx_match[0]
                    result["idx"] = [int(i.strip()) for i in re.findall(r'\d+', idx_str)]
                except:
                    result["idx"] = []
            else:
                premise_nums = re.findall(r'premise\s+(\d+)', response, re.IGNORECASE)
                if premise_nums:
                    result["idx"] = list(set([int(num) for num in premise_nums]))
                else:
                    result["idx"] = []
            if explanation_match:
                result["explanation"] = explanation_match.group(1).strip()
            else:
                result["explanation"] = response[:200] + "..." if len(response) > 200 else response
            return result
        return None

    def run(self, premises, question, tokenizer, model, trace=False):
        prompt = self.generate_prompt(premises, question)
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=1000)
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            input_text_len = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
            response = full_text[input_text_len:].strip()
        parsed = self.parse_response(response)
        result = {
            'question': question,
            'answer': '',
            'idx': [],
            'explanation': '',
            'res': response,
            'error': '',
        }
        if parsed:
            result['answer'] = parsed.get('answer', '')
            idx_data = parsed.get('idx', [])
            if isinstance(idx_data, list):
                result['idx'] = list(set([int(idx) for idx in idx_data if str(idx).isdigit()]))
            elif isinstance(idx_data, str):
                result['idx'] = list(set([int(i.strip()) for i in idx_data.split(',') if i.strip().isdigit()]))
            else:
                result['idx'] = []
            result['explanation'] = parsed.get('explanation', '')
        else:
            result['error'] = 'Failed to parse model response.'
            result['explanation'] = response[:200] + "..." if len(response) > 200 else response
        return result