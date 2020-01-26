from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
fps = FPS().start()

while True:
    
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=9,
                                      minSize=(40,40), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    
    fps.update()
    

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()