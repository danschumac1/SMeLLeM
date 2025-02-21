'''
python ./src/simple_main.py
'''

import json
import os

import numpy as np
from utils.data_management import Prompter, TSDataGenerator
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score


def single_prompt(
        start, end, slope, intercept, noise, num_points, 
         gen_graph=False, save_prompt=False,low_thresh=-0.2, high_thresh=0.2, 
         model="gpt-4o-mini", temperature= .2, verbose=False):
    prompter = Prompter(
        model=model,
        temperature=temperature
    )
    data_gen = TSDataGenerator(
        low_thresh= low_thresh,
        high_thresh= high_thresh,
        num_points= num_points
    )
    data_gen.create_time_series_data(
        slope=slope,
        intercept=intercept,
        noise=noise,
        start=start,
        end=end,
        num_points=num_points,
        gen_graph=gen_graph
    )
    trend = prompter.analyze_trend(
        data_gen.data_str,
        save_prompt=save_prompt
        )
    if not trend:
        print("Error: No trend detected.")
        return None, None
    print(f"Predicted Trend: {trend.direction} \t Ground Truth: {data_gen.label}")
    output_dict = {
        'prediction': trend.direction,
        'ground_truth': data_gen.label,
        'slope': slope,
        'intercept': intercept,
        'noise': noise,
        'num_points': num_points,
        'start': start,
        'end': end,
    }

    with open('./data/results.json', 'a') as f:
        json.dump(output_dict, f)
        f.write('\n')

    return trend.direction, data_gen.label


def evaluate_model():
    """Evaluate the model using accuracy and F1 score."""
    
    # Load results from JSON file
    with open('./data/results.json', 'r') as f:
        results = [json.loads(line) for line in f]

    # Extract true labels and predictions
    true_labels = [result['ground_truth'] for result in results]
    predicted_labels = [result['prediction'] for result in results]

    # Compute Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Compute F1 Score (weighted handles imbalanced data)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")

    return accuracy, f1



    # f1

def randomize_params():
    start = datetime.now().isoformat()
    end = datetime.now().isoformat()
    slope = np.random.uniform(-3, 3)
    intercept = np.random.uniform(-10, 10)
    noise = np.random.uniform(0, 1)
    num_points = np.random.randint(10, 500)
    return start, end, slope, intercept, noise, num_points

def main():
    for i in range(20):
        start, end, slope, intercept, noise, num_points = randomize_params()
        gen_graph = True if i == 0 else False
        save_prompt = True if i == 0 else False
        single_prompt(
            start, end, slope, intercept, noise, num_points, gen_graph, save_prompt)
    evaluate_model()
if __name__ == '__main__':
    main()
