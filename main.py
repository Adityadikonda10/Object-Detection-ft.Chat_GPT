import cv2
import numpy as np
from utils import get_output_layers

# Load YOLOv4 model
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# Load class names
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load image
image = cv2.imread('image.jpg')  # Replace 'image.jpg' with the path to your image

# Preprocess the image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (608, 608), swapRB=True, crop=False)
net.setInput(blob)

# Get the output layer names
output_layers = get_output_layers(net)

# Forward pass
outs = net.forward(output_layers)

# Process the detections
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Assuming you have boxes and confidences from your code
boxes = np.array(boxes)
confidences = np.array(confidences)

# Set the NMS parameters
nms_threshold = 0.5  # You can adjust this value based on your needs
score_threshold = 0.5  # You can adjust this value based on your needs

# Use cv2.dnn.NMSBoxes to perform NMS
indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), score_threshold, nms_threshold)

# Now, indices contain the indices of the bounding boxes to keep after NMS
# You can use these indices to access the filtered boxes and confidences
filtered_boxes = [boxes[i[0]] for i in indices]
filtered_confidences = [confidences[i[0]] for i in indices]

# Draw the bounding boxes after NMS
colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in indices:
    i = i[0]
    box = boxes[i]
    x, y, w, h = box
    label = str(classes[class_ids[i]])
    color = colors[class_ids[i]]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save or display the result
cv2.imwrite('output.jpg', image)  # Save the result to 'output.jpg'
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
