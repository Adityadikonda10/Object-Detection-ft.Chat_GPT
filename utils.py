import cv2
import numpy as np


def get_output_layers(net):
    # Get the names of all layers in the network
    layer_names = net.getLayerNames()

    # Define the indices of the output layers for YOLOv3-320
    output_layer_indices = [-4, -1, 61]

    # Convert indices to layer names
    output_layers = [layer_names[i] for i in output_layer_indices]

    return output_layers



def draw_detection_boxes(frame, outs, classes):
    frame_height, frame_width = frame.shape[:2]
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                x = center_x - width // 2
                y = center_y - height // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indices:
            x, y, width, height = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return boxes

