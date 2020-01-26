from objcenter import ObjCenter
from pid import PID
from multiprocessing import Manager
from multiprocessing import Process
import imutils
from imutils.video import VideoStream
#import pantilthat as pth
from adafruit_servokit import ServoKit
import argparse
import signal
import time
import sys
import cv2


kit = ServoKit(channels=16)

#servoRange = (-90, 90)
panRange = (30, 120)
tltRange = (90, 150)


WIDTH = 320
HEIGHT = 240

iniPan = 75
iniTlt = 120

def signal_handler(sig, frame):
    
    print("[INFO] You pressed 'Ctrl + C'. Exiting...")
    kit.servo[0].angle = iniPan    # Pan
    kit.servo[1].angle = iniTlt    # Tilt      
    sys.exit()
    
    
def obj_center(args, objX, objY, centerX, centerY):
    signal.signal(signal.SIGINT, signal_handler)
    obj = ObjCenter(args["cascade"])
    
    vs = VideoStream(src=0, resolution=(320, 240), framerate=16).start()
    time.sleep(2.0)    # wait for the camera to warm up
    
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=WIDTH)
        frame = cv2.flip(frame, 0)
        
        (H, W) = frame.shape[:2]
        centerX.value = W // 2
        centerY.value = H // 2
        
        objectLoc = obj.update(frame, (centerX.value, centerY.value))
        ((objX.value, objY.value), rect) = objectLoc
        
        if rect is not None:
            (x, y, w, h) = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (255, 0, 255), 2)
            #cv2.arrowedLine(frame, (centerX.value, centerY.value), (objX.value, objY.value), (255, 0, 255), 2)
            cv2.putText(frame, "{}, {}".format(objX.value - centerX.value, objY.value - centerY.value),
                       (centerX.value, centerY.value + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))          
        

        cv2.namedWindow("Pan-Tilt Face Tracking", 0)
        cv2.resizeWindow("Pan-Tilt Face Tracking", 640, 480)
        cv2.imshow("Pan-Tilt Face Tracking", frame)
        cv2.waitKey(1) & 0xFF
    
    
def pid_process(output, p, i, d, objCoord, centerCoord):
    
    signal.signal(signal.SIGINT, signal_handler)
    pid_ctrl = PID(p.value, i.value, d.value)
    pid_ctrl.initialize()
    
    while True:
        error = centerCoord.value - objCoord.value
        output.value = pid_ctrl.update(error)
        

def in_range(val, start, end):
    return (val >= start and val <= end)


def set_servos(pan, tlt):
    signal.signal(signal.SIGINT, signal_handler)
    
    while True:
        panAngle = -1 * pan.value
        tltAngle = -1 * tlt.value
        
        #curPanAngle = kit.servo[0].angle
        #curTltAngle = kit.servo[1].angle
        
        panTarget = iniPan + panAngle
        tltTarget = iniTlt + tltAngle
        
        
        if in_range(panTarget, panRange[0], panRange[1]):
            # print("Pan >>> {:.2f}".format(panTarget))
            kit.servo[0].angle = int(panTarget)

        if in_range(tltTarget, tltRange[0], tltRange[1]):
            # print("Tlt >>> {:.2f}".format(tltTarget))
            kit.servo[1].angle = int(tltTarget)
            

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cascade", type=str, required=True,
                    help="path to input Haar cascade for face detection")
    args = vars(ap.parse_args())
    
    with Manager() as manager:
        
        centerX = manager.Value("i", WIDTH//2)
        centerY = manager.Value("i", HEIGHT//2)
        
        objX = manager.Value("i", WIDTH//2)
        objY = manager.Value("i", HEIGHT//2)
        
        pan = manager.Value("i", 0)
        tlt = manager.Value("i", 0)
        
        panP = manager.Value("f", 0.08)
        panI = manager.Value("f", 0.08)
        panD = manager.Value("f", 0.002)
        
        tltP = manager.Value("f", 0.08)
        tltI = manager.Value("f", 0.08)
        tltD = manager.Value("f", 0.002)
        
        processObjectCenter = Process(target=obj_center,
                                      args=(args, objX, objY, centerX, centerY))
        processPanning = Process(target=pid_process,
                                 args=(pan, panP, panI, panD, objX, centerX))
        processTilting = Process(target=pid_process,
                                 args=(tlt, tltP, tltI, tltD, objY, centerY))
        processSetServos = Process(target=set_servos, args=(pan, tlt))
        
        processObjectCenter.start()
        processPanning.start()
        processTilting.start()
        processSetServos.start()
        
        processObjectCenter.join()
        processPanning.join()
        processTilting.join()
        processSetServos.join()
        
