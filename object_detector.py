from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image
import json

app = Flask(__name__)

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler of /detect POST endpoint
    Receives uploaded file with a name "image_file", 
    passes it through YOLOv8 object detection 
    network and returns an array of bounding boxes.
    :return: a JSON array of objects bounding 
    boxes in format 
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = request.files["image_file"]  # Get the uploaded file
    boxes = detect_objects_on_image(Image.open(buf.stream))  # Detect objects in the image
    return jsonify(boxes)  # Return the bounding boxes as JSON


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format 
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    model = YOLO("best.pt")  # Initialize YOLOv8 model with pre-trained weights
    results = model.predict(buf)  # Use the model to predict objects in the image
    result = results[0]  # Get the results for the first image (assuming batch size of 1)
    output = []  # Initialize an empty list to store the bounding boxes

    for box in result.boxes:  # Iterate through detected objects
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]  # Get coordinates of the bounding box
        class_id = box.cls[0].item()  # Get the class ID of the detected object
        prob = round(box.conf[0].item(), 2)  # Get the confidence score of the detection
        output.append([x1, y1, x2, y2, result.names[class_id], prob])  # Add bounding box information to the output

    return output  # Return the list of bounding boxes

# Start the Flask app with the Waitress server
serve(app, host='0.0.0.0', port=8080)
