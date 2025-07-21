import jsonlines
import math
from utils.tool import *
from utils.prompt import *




def calculate_accuracy(file_path):
    correct = 0
    total = 0
    line = -1

    with jsonlines.open(file_path) as reader:
        for item in reader:
            line += 1
            if line == 0:
                continue
            # print(item)
            total += 1
            question = item['question']
            answer = item['answer']
            generated_code = item['generated'][0]
            prediction = item['executed']


            # executed_result = safe_execute(generated_code)
            # prediction = floatify_ans(executed_result)

            if finqa_equal(prediction, answer):
                correct += 1
            else:
                print(f"Question: {question}")
                print(f"Expected: {answer}, Got: {prediction}")
                print(f"Generated code:\n{generated_code}")
                print("---")

            

    accuracy = correct / total if total > 0 else 0
    print(f"Total questions: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    file_path = "./outputs/gsm8k/test_outputs/gsm8k_vanilla_Llama-3.1-Instruct-8B_tp0.6_topp0.9_s0_e1319_01_06_14_08_seed4.jsonl"
    calculate_accuracy(file_path)