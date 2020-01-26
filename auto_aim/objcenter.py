import cv2

class ObjCenter:
    def __init__(self, haarPath):
        self.detector = cv2.CascadeClassifier(haarPath)
        
    def update(self, frame, frameCenter):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.05,
                                               minNeighbors=9, minSize=(30,30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)
        
        # if a face was found
        if len(rects) > 0:
            (x, y, w, h) = rects[0]
            faceX = int(x + (w / 2.0))
            faceY = int(y + (h / 2.0))
            
            return ((faceX, faceY), rects[0])
        
        # no faces were found
        return (frameCenter, None)
    
    
    