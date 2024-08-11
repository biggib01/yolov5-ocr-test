import cv2

def perform_yolo_ocr_on_area(image_path, model, confidence_threshold=0.6):
    # Load the image
    img = cv2.imread(image_path)

    # Perform text detection using the YOLO model
    results = model(image_path)

    # Initialize variables to store detections and warnings
    detections = []
    warnings = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        labels = [model.names[class_id] for class_id in class_ids]

        for i, (box, label, confidence) in enumerate(zip(boxes, labels, confidences)):
            if confidence < confidence_threshold:
                warnings.append({i})

            detection = {
                'index': i,
                'class': label,
                'confidence': float(confidence)
            }
            detections.append(detection)

    # Combine results into a final output
    if warnings:
        return {
            'detections': detections,
            'warnings': warnings
        }
    elif not detections:
        return "No detections found."
    else:
        return {
            'detections': detections,
            'warnings': []  # No warnings if all confidences are acceptable
        }