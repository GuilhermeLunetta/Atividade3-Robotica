# MOBILENET_DETECTION
# Use o projeto mobilenet_detection para basear seu código.
#
#Neste projeto, escolha uma categoria de objetos que o reconhecedor reconhece. Diga aqui qual foi sua escolha
#
#Implemente a seguinte funcionalidade: sempre que o objeto identificado em (2) estiver presente por mais que 5 frames seguidos, desenhe um retângulo fixo ao redor dele.

# Para RODAR
# python3 atividade3ex3.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# Importing important libraries
import numpy as np
import cv2
import argparse
import time

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Categories
CLASSES = ["person"]

# Confidence variable and a list of color to paint the rectangles
CONFIDENCE = 0.7
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Neural network initialization
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Detect function
def detect(frame):
    " Recebe: uma imagem colorida "
    " Devolve: objeto encontrado "
    
    image = frame.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections and predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    results = []

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the "confidence" is greater than the minimum confidence
        if confidence > CONFIDENCE:
            # Extract the index of the class label from the "detections",
            # then compute the (x, y) coordinates of the bounding box for the object
            idx = 0
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            if frames_count >= 5:
               
                # Display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            results.append((CLASSES[idx], confidence*100, (startX, startY),(endX, endY) ))

    # show the output image
    return image, results

# Video capture
cap = cv2.VideoCapture(0)

frames_count = 0

while True:
    #Capture frame by frame
    ret, frame = cap.read()

    result_frame, result_tuple = detect(frame)

    #Text to quit
    cv2.putText(result_frame, "Press Q to Quit", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    #Display the resulting frame
    cv2.imshow("Detector", result_frame)

    #Print the tuples
    for tuple in result_tuple:
        if tuple[0] == 'person':
            if frames_count <= 5:
                frames_count += 1
        print(tuple)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()