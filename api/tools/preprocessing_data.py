import json
import os

def split_json_questions(input_json_path, output_json_path):
    """
    Split questions in a JSON file into separate dictionaries for each question.
    
    Args:
        input_json_path (str): Path to the input JSON file.
        output_json_path (str): Path to save the output JSON file.
    """
    # Read the input JSON file
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize the new list for split entries
    new_data = []
    
    # Process each item in the input data
    for item in data:
        premises_nl = item.get('premises-NL', [])
        premises_fol = item.get('premises-FOL', [])
        questions = item.get('questions', [])
        answers = item.get('answers', [])
        idx = item.get('idx', [])
        explanations = item.get('explanation', [])
        
        # Ensure the lengths match
        if not (len(questions) == len(answers) == len(idx) == len(explanations)):
            raise ValueError(f"Mismatch in lengths for item: {item}")
        
        # Create a new dictionary for each question
        for i, question in enumerate(questions):
            new_item = {
                'premises-NL': premises_nl,  # Keep the full list of premises-NL
                'premises-FOL': premises_fol,  # Keep the full list of premises-FOL
                'question': question,
                'answer': answers[i],
                'idx': idx[i],  # Take the corresponding idx list
                'explanation': explanations[i]
            }
            new_data.append(new_item)
    
    # Write the new data to the output JSON file
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    print(f"Split data saved to {output_json_path}")

if __name__ == "__main__":
    # Example usage
    input_path = "data/train_v1.json"
    output_path = "data/preprocessed_train_v1.json"
    split_json_questions(input_path, output_path)