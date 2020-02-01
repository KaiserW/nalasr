from utils.yolov3_detector import YoloNet
from utils.conf import Conf
from imagezmq import imagezmq
import imutils
import cv2
from flask import Flask, Response, render_template
import threading
import argparse


outputFrame = None
lock = threading.Lock()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace;boundary=frame")


def detect_object():

    global outputFrame, lock

    imageHub = imagezmq.ImageHub()

    # YOLOv3 Tiny network
    print("[INFO] loading YOLO from disk...")
    net = YoloNet(conf)
    print("[INFO] activating sentry turret...")

    while True:
        (rpiName, frame) = imageHub.recv_image()
        imageHub.send_reply(b"OK")

        net.predict(frame)
        frame = net.visualize(rpiName)

        with lock:
            outputFrame = frame.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def generate():

    global outputFrame, lock

    while True:
        with lock:

            if outputFrame is None:
                continue

            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            if not flag:
                continue

        yield b'--frame\r\n' b'ContentType: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True,
                help="Path to the input configuration file")
    args = vars(ap.parse_args())

    # Reading VideoStream from Respberry
    conf = Conf(args["conf"])

    t = threading.Thread(target=detect_object)
    t.daemon = True
    t.start()

    app.run(host="0.0.0.0", port=8000, debug=True,
            threaded=True, use_reloader=False)

cv2.destroyAllWindows()
