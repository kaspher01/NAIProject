from transformers import pipeline
import pandas as pd
from utils.calculator import calculate_metrics


def model_falconsai(dataset_path):
    # Wczytanie zbioru danych
    dataset = pd.read_excel(dataset_path, sheet_name='dataset')
    # Inicjalizacja pipeline do sumaryzacji
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")

    # Inicjalizacja listy do przechowywania metryk dla wszystkich rekordów
    all_precisions, all_recalls, all_f1s = [], [], []

    for index, row in dataset.iterrows():
        article = row['article']
        actual_summary = summarizer(article, max_length=230, min_length=30, do_sample=False)[0]['summary_text']
        predicted_summary = row['predicted_summary']

        # Użycie funkcji calculate_metrics
        # Obliczenie precyzji, recall i f1
        precision, recall, f1 = calculate_metrics(predicted_summary, actual_summary)

        # Dodanie metryki do list
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    # Obliczenie średniej metryki dla całego zbioru danych
    model2_precision = sum(all_precisions) / len(all_precisions)
    model2_recall = sum(all_recalls) / len(all_recalls)
    model2_f1 = sum(all_f1s) / len(all_f1s)
    return model2_precision, model2_recall, model2_f1
