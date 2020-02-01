from utils.parse_yolo_output import ParseYOLOOutput
import os
import numpy as np
import cv2


class YoloNet:

    def __init__(self, conf):

        self.conf = conf
        self.labelsPath = os.path.sep.join([conf["yolo_path"], "coco.names"])
        self.LABELS = open(self.labelsPath).read().strip().split("\n")

        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")

        self.weightsPath = os.path.sep.join([conf["yolo_path"], "yolov3-tiny.weights"])
        self.configPath = os.path.sep.join([conf["yolo_path"], "yolov3-tiny.cfg"])

        # Object Detector
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        ln = self.net.getLayerNames()
        self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.pyo = ParseYOLOOutput(conf)

        self.name = None
        self.frame = None

        self.layerOutputs = None
        self.boxes = None
        self.confidences = None
        self.classIDs = None
        self.idxs = None

    def predict(self, frame):

        self.frame = frame

        H, W = self.frame.shape[:2]

        blob = cv2.dnn.blobFromImage(self.frame, 1 / 255.0, (288, 288), swapRB=True, crop=False)
        self.net.setInput(blob)
        self.layerOutputs = self.net.forward(self.ln)

        (self.boxes, self.confidences, self.classIDs) = self.pyo.parse(self.layerOutputs, self.LABELS, H, W)

        self.idxs = cv2.dnn.NMSBoxes(self.boxes, 
                                     self.confidences, 
                                     self.conf["confidence"], 
                                     self.conf["threshold"])

    def visualize(self, name):

        if len(self.idxs) > 0:

            for i in self.idxs.flatten():
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])

                color = [int(c) for c in self.COLORS[self.classIDs[i]]]
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
                
                text = "{}: {:.2f}".format(self.LABELS[self.classIDs[i]], self.confidences[i])
                y = (y - 15) if (y - 15) > 0 else h - 15
                cv2.putText(self.frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ''' showing on web instead of window
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 640, 480)
        cv2.imshow(name, self.frame)
        '''

        return self.frame
