from ultralyticsplus import YOLO, render_result


def visualize_object_detection_model3(file_path):
    # Load model
    model = YOLO('ultralyticsplus/yolov8s')

    # Set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image

    # Perform inference
    results = model.predict(file_path)

    # Observe results
    print(results[0].boxes)

    # Render and display the result
    render = render_result(model=model, image=file_path, result=results[0])
    render.show()