from transformers import pipeline
import pandas as pd
from utils.calculator import calculate_metrics

def model_bart_large_cnn(dataset_path):
    # Wczytanie zbioru danych
    dataset = pd.read_excel(dataset_path, sheet_name='dataset')

    # Inicjalizacja pipeline do sumaryzacji
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Inicjalizacja listy do przechowywania metryk dla wszystkich rekordów
    all_precisions, all_recalls, all_f1s = [], [], []

    for index, row in dataset.iterrows():
        # Pobieranie rzeczywistej sumaryzacji
        actual_summary = summarizer(row['article'], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        # Pobieranie przewidzianej sumaryzacji
        predicted_summary = row['predicted_summary']

        # Użycie funkcji calculate_metrics
        # Obliczanie precyzji, recall i f1
        precision, recall, f1 = calculate_metrics(predicted_summary, actual_summary)

        # Dodanie metryki do list
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    # Obliczanie średniej metryki dla całego zbioru danych
    model1_precision = sum(all_precisions) / len(all_precisions)
    model1_recall = sum(all_recalls) / len(all_recalls)
    model1_f1 = sum(all_f1s) / len(all_f1s)

    return model1_precision, model1_recall, model1_f1
