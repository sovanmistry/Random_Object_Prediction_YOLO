#!/usr/bin/env python
# coding: utf-8

# ## Import Necessary Libraries

import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader


class YOLO_pred():
    
    def __init__(self, onnx_model, data_yaml):
        
        # ## STEP 1 : LOAD YAML FILE
        with open('data.yaml', mode = 'r') as f:
            data_yaml = yaml.load(f, Loader = SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']


        # ## STEP2 : LOAD YOLO MODEL with OPENCV       
        self.yolo = cv2.dnn.readNetFromONNX('./Model5/weights/best.onnx')
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        
        
    def predictions(self, image):

        # Check Height Width and Depth of image
        row, col, d = image.shape

        # ### TASK : Covert our image into Square Image
        # 1. decide the larger between row, col
        max_rc = max(row, col)

        # 2. Black Shell Matrix
        input_image = np.zeros((max_rc, max_rc, 3), dtype = np.uint8)

        # 3. Fill the Black Matrix with our Image
        input_image [0:row, 0:col] = image


        ## STEP 4 : GET PREDICTIONS FROM SQUARE ARRAY

        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image , 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop= False)
        self.yolo.setInput(blob)

        # Prediction from YOLO
        preds = self.yolo.forward() 

        # ## STEP 5:  NON MAX SUPPRESSION 

        # ### Filter Detection based on Confidence 0.4 and Probability Score 0.25

        detections = preds[0]

        boxes = []
        confidences = []
        classes = []


        # Caclculate Width and Height of the Image
        image_w, image_h = input_image.shape[:2]

        # Calculate X_Factor & Y_Factor
        x_factor = image_w/INPUT_WH_YOLO
        y_factor = image_h/INPUT_WH_YOLO

        # Filter Detection based on Confidence 0.4 and Probability Score 0.25

        for i in range(len(detections)):
            row = detections[i]

            # Confidence of detecting an Object
            confidence = row[4] 

            if confidence > 0.4: 

                # Maximum Probability from 20 Objects
                class_score = row[5:].max()

                # Which Class gives Max Probablitiy
                class_id = row[5:].argmax()

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]

                    # CONSTRUCT BOUNDING BOXES from the values
                    # Left, TOP, WIDTH & HEIGHT

                    left   = int((cx - 0.5*w)* x_factor)
                    top    = int((cy - 0.5*h)* y_factor)
                    width  = int(w*x_factor)
                    height = int(h*y_factor)

                    box = np.array([left, top, width, height])

                    # Append Values into List
                    confidences.append (confidence)
                    boxes.append (box)
                    classes.append (class_id)



        # ### Clean Duplicate Values & Store it in Numpy List 

        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()


        # ### Non Max Suppression Operation
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()

        for ind in index : 

            # Extract Bounding Box
            x, y, w, h = boxes_np[ind]
            bb_conf = confidences_np[ind] * 100
            classes_id = classes[ind]
            class_name = self.labels[classes_id]
            colors = self.generate_colors(classes_id)
            

            text = f'{class_name} : {bb_conf}%'

            cv2.rectangle(image, (x,y), (x+w, y+h), colors, 2)
            cv2.rectangle(image, (x,y-30), (x+w, y), colors, -1)

            cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0),1)
        
        return image
    
    
    def generate_colors(self, ID):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size = (self.nc, 3)).tolist()
        return tuple(colors[ID])

        # ## THE END
