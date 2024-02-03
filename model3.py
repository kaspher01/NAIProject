from transformers import pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

def calculate_metrics(predicted_summary, actual_summary):
    # Convert predicted and actual summaries to binary arrays
    predicted_bin = [1 if word in predicted_summary else 0 for word in actual_summary]
    actual_bin = [1 if word in actual_summary else 0 for word in actual_summary]

    # Calculate metrics using scikit-learn functions
    precision = precision_score(actual_bin, predicted_bin)
    recall = recall_score(actual_bin, predicted_bin)
    f1 = f1_score(actual_bin, predicted_bin)

    return precision, recall, f1

def model3(dataset_path):
    # Load the dataset
    dataset = pd.read_excel(dataset_path, sheet_name='dataset')

    # Initialize summarizer pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Initialize lists to store metrics for all records
    all_precisions, all_recalls, all_f1s = [], [], []

    for index, row in dataset.iterrows():
        actual_summary = summarizer(row['article'], max_length=150, min_length=30, do_sample=False)
        predicted_summary = row['predicted_summary']

        actual_summary_text = actual_summary[0]['summary_text']
        actual_summary_words = actual_summary_text.split()

        # Use the calculate_metrics function
        precision, recall, f1 = calculate_metrics(predicted_summary.split(), actual_summary_words)

        # Append metrics to lists
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    # Calculate mean metrics for the entire dataset
    mean_precision = sum(all_precisions) / len(all_precisions)
    mean_recall = sum(all_recalls) / len(all_recalls)
    mean_f1 = sum(all_f1s) / len(all_f1s)

    print("Model 3 - pszemraj/led-base-book-summary")
    print(f"Precision: {mean_precision}, Recall: {mean_recall}, F1: {mean_f1}")

# Call the function with the dataset path
model3('dataset.xlsx')  # Change 'your_dataset.xlsx' to your actual dataset path
