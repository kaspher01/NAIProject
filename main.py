from models.model_bart_large_cnn import model_bart_large_cnn
from models.model_falconsai import model_falconsai
from models.model_pszemraj_book_summary import model_pszemraj_book_summary
from utils.printer import print_all_results

if __name__ == "__main__":
    dataset_path = "data/dataset.xlsx"
    print("Loading first model...")
    model1_metrics = (model1_precision, model1_recall, model1_f1) = model_bart_large_cnn(dataset_path)
    print("Loading second model...")
    model2_metrics = (model2_precision, model2_recall, model2_f1) = model_falconsai(dataset_path)
    print("Loading third model...")
    model3_metrics = (model3_precision, model3_recall, model3_f1) = model_pszemraj_book_summary(dataset_path)
    print_all_results(model1_metrics, model2_metrics, model3_metrics)
