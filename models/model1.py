from transformers import pipeline
import pandas as pd
from utils.calculator import calculate_metrics


def model_bart_large_cnn(dataset_path):
    # Load the dataset
    dataset = pd.read_excel(dataset_path, sheet_name='dataset')

    # Initialize summarizer pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Initialize lists to store metrics for all records
    all_precisions, all_recalls, all_f1s = [], [], []

    for index, row in dataset.iterrows():
        actual_summary = summarizer(row['article'], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        predicted_summary = row['predicted_summary']

        # Use the calculate_metrics function
        precision, recall, f1 = calculate_metrics(predicted_summary, actual_summary)

        # Append metrics to lists
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    # Calculate mean metrics for the entire dataset
    model1_precision = sum(all_precisions) / len(all_precisions)
    model1_recall = sum(all_recalls) / len(all_recalls)
    model1_f1 = sum(all_f1s) / len(all_f1s)

    return model1_precision, model1_recall, model1_f1
