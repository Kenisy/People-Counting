# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import tensorflow as tf
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="models/ssd_mobilenet_v2.pb",
	help="path to Tennsorflow pre-trained model")
ap.add_argument("-i", "--input", type=str, default="input/input.mp4",
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str, default="output/output.mp4",
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=2,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
# Read the graph.
with tf.gfile.FastGFile(args["model"], 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# door area [x1, y1, x2, y2]
door = [-1] * 4
# outer door area
outer_door = [-1] * 4

# start the frames per second throughput estimator
fps = FPS().start()

def IsInRect(position, rect):
	if position[0] >= rect[0] and position[0] <= rect[2] and position[1] >= rect[1] and position[1] <= rect[3]:
		return True
	return False


with tf.Session() as sess:
	# Restore session
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')

	# loop over frames from the video stream
	while True:
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame

		# if we are viewing a video and we did not grab a frame then we
		# have reached the end of the video
		if args["input"] is not None and frame is None:
			break

		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		frame = imutils.resize(frame, width=500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them and door area
		if W is None or H is None:
			(H, W) = frame.shape[:2]
			door[0] = W // 2 - 50
			door[1] = H // 2 + 40
			door[2] = W // 2 + 50
			door[3] = H - 20
			outer_door[0] = door[0] - 50
			outer_door[1] = door[1] - 50
			outer_door[2] = door[2] + 50
			outer_door[3] = door[3]

		# if we are supposed to be writing a video to disk, initialize
		# the writer
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MP4V")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)

		# initialize the current status along with our list of bounding
		# box rectangles returned by either (1) our object detector or
		# (2) the correlation trackers
		status = "Waiting"
		rects = []

		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if totalFrames % args["skip_frames"] == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []

			# Read and preprocess an image.
			rows = frame.shape[0]
			cols = frame.shape[1]
			if args["model"] == "models/ssd_mobilenet_v2.pb":
				inp = cv2.resize(frame, (300, 300))
			else:
				inp = cv2.resize(frame, (H, W))
			inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

			# Run the model
			out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
							sess.graph.get_tensor_by_name('detection_scores:0'),
							sess.graph.get_tensor_by_name('detection_boxes:0'),
							sess.graph.get_tensor_by_name('detection_classes:0')],
						feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

			# Visualize detected bounding boxes.
			detections = int(out[0][0])

			# loop over the detections
			for i in range(detections):
				# extract the confidence (i.e., probability) associated
				# with the prediction
				confidence = float(out[1][0][i])

				# filter out weak detections by requiring a minimum
				# confidence
				if confidence > args["confidence"]:
					# extract the index of the class label from the
					# detections list
					# idx = int(detections[0, 0, i, 1])

					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = [float(v) for v in out[2][0][i]]
					startX = int(box[1] * cols)
					startY = int(box[0] * rows)
					endX = int(box[3] * cols)
					endY = int(box[2] * rows)

					# construct a dlib rectangle object from the bounding
					# box coordinates and then start the dlib correlation
					# tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					# add the tracker to our list of trackers so we can
					# utilize it during skip frames
					trackers.append(tracker)

		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"

				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

		# draw a horizontal line in the center of the frame -- once an
		# object crosses this line we will determine whether they were
		# moving 'up' or 'down'
		cv2.rectangle(frame, (door[0], door[1]), (door[2], door[3]), (0, 255, 255), 1)
		cv2.rectangle(frame, (outer_door[0], outer_door[1]), (outer_door[2], outer_door[3]), (0, 0, 255), 1)

		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects, disappeared = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the y-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'up' and positive for 'down')
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# check if object in door area
				if to.wasInDoor == False and IsInRect(centroid, door):
					to.wasInDoor = True

				# check to see if the object has been counted or not
				if not to.counted and to.wasInDoor:
					# if object was in door and disappeared or move down from the door area count object down
					if disappeared[objectID] > 30 or (direction > 0 and centroid[1] > door[3]):
						totalDown += 1
						to.counted = True

					# if object was in door and move out of outer door count object up
					elif not IsInRect(centroid, outer_door):
						totalUp += 1
						to.counted = True

			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		# construct a tuple of information we will be displaying on the
		# frame
		info = [
			("Up", totalUp),
			("Down", totalDown),
			("Frame", totalFrames),
			("Status", status),
		]

		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		# check to see if we should write the frame to disk
		if writer is not None:
			writer.write(frame)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# increment the total number of frames processed thus far and
		# then update the FPS counter
		totalFrames += 1
		fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()