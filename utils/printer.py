def print_results(model_name, precision, recall, f1):
    rounded_precision = "{:.2f}".format(precision)
    rounded_recall = "{:.2f}".format(recall)
    rounded_f1 = "{:.2f}".format(f1)

    print(f"{model_name} - Metrics:")
    print(f"Precision: {rounded_precision}, Recall: {rounded_recall}, F1: {rounded_f1}\n")


def print_all_results(model1_metrics, model2_metrics, model3_metrics):
    print("Models Metrics:\n")
    print_results("Model 1 - facebook/bart-large-cnn", *model1_metrics)
    print_results("Model 2 - Falconsai/text_summarization", *model2_metrics)
    print_results("Model 3 - pszemraj/led-base-book-summary", *model3_metrics)
