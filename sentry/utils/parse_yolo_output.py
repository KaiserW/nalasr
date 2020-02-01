# import the necessary package
import numpy as np


class ParseYOLOOutput:
	def __init__(self, conf):
		# store the configuration file
		self.conf = conf

	def parse(self, layerOutputs, LABELS, H, W):
		# initialize our lists of detected bounding boxes,
		# confidences, and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID
				scores = detection[5:]
				classID = np.argmax(scores)

				# check if the class detected should be considered,
				# if not, then skip this iteration
				if LABELS[classID] not in self.conf["consider"]:
					continue

				# retrieve the confidence (i.e., probability) of the
				# current object detection
				confidence = scores[classID]

				# filter out weak predictions by ensuring the
				# detected probability is greater than the minimum
				# probability
				if confidence > self.conf["confidence"]:
					# scale the bounding box coordinates back
					# relative to the size of the image, keeping in
					# mind that YOLO actually returns the center
					# (x, y)-coordinates of the bounding box followed
					# by the boxes' width and height
					box = detection[0:4] * np.array([W, H, W, H])
					box = box.astype("int")
					(centerX, centerY, width, height) = box

					# use the center (x, y)-coordinates to derive the
					# top and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates,
					# confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# return the detected boxes and their corresponding
		# confidences and class IDs
		return (boxes, confidences, classIDs)