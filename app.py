from flask import Flask, request, send_from_directory, url_for

from flask_cors import CORS
import base64
import cv2

import model

from logging.config import dictConfig

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "wsgi": {
                "class": "logging.StreamHandler",
                "stream": "ext://flask.logging.wsgi_errors_stream",
                "formatter": "default",
            }
        },
        "root": {"level": "INFO", "handlers": ["wsgi"]},
    }
)

app = Flask("swans-identification-backend")
CORS(app)
app.config["IMAGES_TO_PROCESS"] = "images/to_process/"
app.config["SAVED_IMAGES"] = "images/saved/"


@app.route("/segment", methods=["POST"])
def segment():
    app.logger.info(f"Got files {[x.filename for x in request.files.getlist('f[]')]}")
    # for file in request.files.getlist("f[]"):
    file = request.files.getlist("f[]")[0]
    mask = model.segment(file.stream.read())

    img = cv2.imencode(".png", mask)[1].tobytes()
    return {
        "file_name": f"mask-{file.filename}",
        "file_type": file.mimetype,
        "data": base64.b64encode(img).decode(),
    }


if __name__ == "__main__":
    app.run("0.0.0.0", 5000)
