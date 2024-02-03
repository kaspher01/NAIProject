from transformers import pipeline
import pandas as pd
from utils.calculator import calculate_metrics


def model_falconsai(dataset_path):
    dataset = pd.read_excel(dataset_path, sheet_name='dataset')
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")

    # Initialize lists to store metrics for all records
    all_precisions, all_recalls, all_f1s = [], [], []

    for index, row in dataset.iterrows():
        article = row['article']
        actual_summary = summarizer(article, max_length=230, min_length=30, do_sample=False)[0]['summary_text']
        predicted_summary = row['predicted_summary']

        # Use the calculate_rouge_metrics function
        precision, recall, f1 = calculate_metrics(predicted_summary, actual_summary)

        # Append metrics to lists
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    # Calculate mean metrics for the entire dataset
    model2_precision = sum(all_precisions) / len(all_precisions)
    model2_recall = sum(all_recalls) / len(all_recalls)
    model2_f1 = sum(all_f1s) / len(all_f1s)
    return model2_precision, model2_recall, model2_f1
