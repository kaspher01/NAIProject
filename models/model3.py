from transformers import pipeline
import pandas as pd
import torch
from utils.calculator import calculate_metrics


def model_pszemraj_book_summary(dataset_path):
    # Load the dataset
    dataset = pd.read_excel(dataset_path, sheet_name='dataset')

    # Initialize summarizer pipeline
    hf_name = "pszemraj/led-base-book-summary"
    summarizer = pipeline(
        "summarization",
        hf_name,
        device=0 if torch.cuda.is_available() else -1,
    )

    # Initialize lists to store metrics for all records
    all_precisions, all_recalls, all_f1s = [], [], []

    for index, row in dataset.iterrows():
        # Assuming you have a 'predicted_summary' column in your dataset
        predicted_summary = row['predicted_summary']

        actual_summary_results = summarizer(row['article'],
                                            min_length=8,
                                            max_length=256,
                                            no_repeat_ngram_size=3,
                                            encoder_no_repeat_ngram_size=3,
                                            repetition_penalty=3.5,
                                            num_beams=4,
                                            do_sample=False,
                                            early_stopping=True,
                                            )

        # Extract actual summaries from the summarizer results
        actual_summaries = [result['summary_text'] for result in actual_summary_results]

        # Calculate metrics for each actual summary
        for actual_summary_text in actual_summaries:
            # Use the calculate_metrics function
            precision, recall, f1 = calculate_metrics(predicted_summary, actual_summary_text)

            # Append metrics to lists
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)

    # Calculate mean metrics for the entire dataset
    model3_precision = sum(all_precisions) / len(all_precisions)
    model3_recall = sum(all_recalls) / len(all_recalls)
    model3_f1 = sum(all_f1s) / len(all_f1s)

    return model3_precision, model3_recall, model3_f1
