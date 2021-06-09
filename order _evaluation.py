import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("weights/yolov4-obj_best_89MAP.weights", "yolov4-obj.cfg")
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
price_list = np.random.uniform(100, 500, size=(len(classes)))

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("images/15144.jpg")
#img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.1:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

sumPrice = 0
indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold=0.05,nms_threshold=0.1,top_k=2)
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        print(classes[class_ids[i]], confidences[i])
        price = round(price_list[class_ids[i]])
        sumPrice += price
        label = str(classes[class_ids[i]]) + " " + str(price) + "RUB"
        color = (0,0,150)#colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x+30, y + 30), font, 0.7, color, 1)
        
labelSummaryPrice = "Total order price: " + str(sumPrice) + "RUB"
cv2.putText(img, labelSummaryPrice, (60, height-20), font, 1, (0,0,0), 1)
#cv2.namedWindow("Image", cv2.WINDOW_NORMAL) 
#cv2.resizeWindow("Image", 600, 600)
cv2.imshow("Total order price", img)
cv2.waitKey(0)
cv2.destroyAllWindows()