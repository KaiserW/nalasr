from utils.yolov3_detector import YoloNet
from utils.conf import Conf
import zmq
from imagezmq import imagezmq
import imutils
import cv2
from flask import Flask, Response, render_template, request
import threading
import argparse


outputFrame = None
lock = threading.Lock()

# HUD server
app = Flask(__name__)

# Listening front-end command
context = zmq.Context()


@app.route("/")
def index():
    return render_template("index.html")


# Reading VideoStream from Respberry
@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace;boundary=frame")


# Receiving mouse click postion
@app.route("/post_coord", methods=["POST"])
def get_coord():

    x = request.form["x"]
    y = request.form["y"]

    print("[INFO] sending command: x={}, y={}".format(x, y))

    socket = context.socket(zmq.REQ)
    socket.connect("tcp://{}:5556".format(args["server_ip"]))

    cmd_dict = {"x": x,
                "y": y}
    socket.send_json(cmd_dict)

    response = socket.recv().decode("ascii")
    print("[INFO] received reply '{}'".format(response))

    return "200 OK"


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

    cv2.destroyAllWindows()


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
    ap.add_argument("-ip", "--server-ip", required=True,
        help="IP of the client (Raspberry)")
    args = vars(ap.parse_args())

    # Reading VideoStream from Respberry
    conf = Conf(args["conf"])

    t = threading.Thread(target=detect_object)
    t.daemon = True
    t.start()

    app.run(host="0.0.0.0", port=8000, debug=True,
            threaded=True, use_reloader=False)

