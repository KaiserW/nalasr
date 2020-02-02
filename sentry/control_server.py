from flask import Flask, render_template, request, Response


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/post_coord", methods=["POST"])
def get_coord():
    x = request.form["x"]
    y = request.form["y"]
    print("x:{}, y:{}".format(x, y))
    return "200 OK"

app.run(host="0.0.0.0", port=8000, debug=True,
            threaded=True, use_reloader=False)