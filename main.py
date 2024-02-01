from model1 import visualize_object_detection_model1
from model2 import visualize_object_detection_model2
from model3 import visualize_object_detection_model3

if __name__ == "__main__":
    file_path = "https://d-art.ppstatic.pl/kadry/k/r/4b/e0/644912102461b_o_full.jpg"
    visualize_object_detection_model1(file_path)
    visualize_object_detection_model2(file_path)
    visualize_object_detection_model3(file_path)
