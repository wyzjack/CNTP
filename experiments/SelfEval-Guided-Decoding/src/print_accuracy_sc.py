import jsonlines
import math
import argparse
from utils.tool import *
from utils.prompt import *

def calculate_accuracy(file_paths):
    correct = 0
    total = 0
    first_file = True
    questions_predictions = {}  # Dictionary to store predictions for each question

    # Process each file and collect predictions
    for file_path in file_paths:
        line = -1
        try:
            with jsonlines.open(file_path) as reader:
                for item in reader:
                    line += 1
                    if line == 0:
                        continue
                    
                    question = item['question']
                    answer = item['answer']
                    prediction = item['executed']
                    # print(prediction)

                    if first_file:
                        questions_predictions[question] = {'answer': answer, 'predictions': []}
                        total += 1
                    
                    questions_predictions[question]['predictions'].append(prediction)

            first_file = False
        except FileNotFoundError:
            print(f"Warning: Could not find file {file_path}")
            continue

    # Calculate accuracy using majority voting
    for question, question_data in questions_predictions.items():
        predictions = question_data['predictions']
        true_answer = question_data['answer']
        
        # Count frequencies of predictions
        pred_freq = {}
        for pred in predictions:
            pred_freq[pred] = pred_freq.get(pred, 0) + 1
        
        # Get majority prediction
        majority_pred = max(pred_freq.items(), key=lambda x: x[1])[0]
        
        if finqa_equal(majority_pred, true_answer):
            correct += 1
        else:
            print(f"Question: {question}")
            print(f"Expected: {true_answer}, Got: {majority_pred}")
            print(f"Prediction distribution: {pred_freq}")
            print("---")

    accuracy = correct / total if total > 0 else 0
    print(f"Total questions: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate accuracy using majority voting from multiple prediction files')
    parser.add_argument('file_pattern', help='File pattern to match prediction files (e.g., "gsm8k_vanilla_DeepSeek-R1-Distill-Qwen-1.5B-1.5B_tp0.6_topp0.95_s0_e1319_*_seed{}_entropy_low0.01_entropy_high1.5_maxtrial10.jsonl")')
    parser.add_argument('--base_path', default='./outputs/gsm8k/test_outputs/', help='Base path for output files (default: ./outputs/gsm8k/test_outputs/)')
    
    args = parser.parse_args()
    
    base_path = args.base_path
    file_pattern = args.file_pattern
    
    import glob
    import re
    
    # Get all matching files
    all_files = glob.glob(base_path + file_pattern.format('*'))
    
    # Sort files by seed number
    def get_seed_number(filepath):
        match = re.search(r'seed(\d+)', filepath)
        return int(match.group(1)) if match else -1
    
    # Filter files to get exactly 40 files with consecutive seed numbers
    sorted_files = sorted(all_files, key=get_seed_number)
    if len(sorted_files) < 40:
        print(f"Warning: Found only {len(sorted_files)} files, expected 40")
    
    file_paths = sorted_files[:40]
    calculate_accuracy(file_paths)