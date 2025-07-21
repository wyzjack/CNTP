import json
import glob
import argparse
from collections import defaultdict, Counter

def load_json_objects(filename):
    """
    Load one or more JSON objects from a file.
    The file can either contain a single JSON object or
    be newline-delimited JSON (NDJSON).
    """
    objects = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            return objects
        try:
            # Try to load the entire file as one JSON object
            data = json.loads(content)
            # If it's a list, extend the list; if it's a dict, wrap it in a list.
            if isinstance(data, list):
                objects.extend(data)
            elif isinstance(data, dict):
                objects.append(data)
            else:
                raise ValueError("JSON content is neither a dict nor a list.")
        except json.JSONDecodeError:
            # If the file is NDJSON, load line by line.
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    objects.append(obj)
    return objects

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze JSON prediction files and calculate majority vote accuracy')
    parser.add_argument('json_pattern', 
                       help='Glob pattern for JSON files (e.g., "./MATH/deepseek-r1-distill-qwen-1.5B_cautious_*/predictions.jsonl")')
    
    args = parser.parse_args()
    
    # Use the provided pattern instead of hardcoded path
    json_files = glob.glob(args.json_pattern)
    print(json_files)
    
    # Dictionary to group runs by question.
    # Key: question text; Value: list of (prediction, answer) tuples.
    question_results = defaultdict(list)
    
    for filename in json_files:
        objects = load_json_objects(filename)
        for obj in objects:
            question = obj.get("question", "").strip()
            prediction = obj.get("prediction", "").strip()
            answer = obj.get("answer", "").strip()
            if question and prediction and answer:
                question_results[question].append((prediction, answer))
            else:
                print(f"Warning: Missing field in file {filename}: {obj}")
    
    total_questions = 0
    correct_questions = 0
    
    # Process each question
    for question, results in question_results.items():
        # All runs for a given question should share the same ground truth answer.
        ground_truth = results[0][1]
        # Get the list of predictions
        predictions = [pred for pred, _ in results]
        # Count votes for each prediction.
        vote_counter = Counter(predictions)
        majority_vote, vote_count = vote_counter.most_common(1)[0]
        total_questions += 1
        is_correct = (majority_vote == ground_truth)
        if is_correct:
            correct_questions += 1
        print(f"Question: {question}")
        print(f"  Ground truth: {ground_truth}")
        print(f"  Majority vote: {majority_vote} (votes: {vote_counter})")
        print(f"  Correct: {is_correct}")
        print("-" * 40)
    
    # Calculate and print final accuracy
    accuracy = (correct_questions / total_questions) * 100 if total_questions > 0 else 0
    print(f"\nFinal accuracy: {accuracy:.2f}% ({correct_questions} out of {total_questions})")
    
if __name__ == "__main__":
    main()
