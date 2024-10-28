from flask import Flask, render_template, request, jsonify # type: ignore
from ultralytics import YOLO # type: ignore
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load the YOLO model
model = YOLO("/mnt/c/Users/Shreyas N/Desktop/Proj/models/best_saved_model/best_float16.tflite")  # Update with your actual model path

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Receive the image data from the request
    try:
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        img_array = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Run YOLO inference
        results = model.predict(frame)
        
        # Ensure results are available
        if results and len(results) > 0:
            annotated_frame = results[0].plot()  # Draw the bounding boxes on the frame
        else:
            print("No detections found.")
            annotated_frame = frame  # Return the original frame if no detections

        # Encode the processed frame back to base64 to send to frontend
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        processed_img = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'processed_image': f'data:image/jpeg;base64,{processed_img}'})
    
    except Exception as e:
        print(f"Error in detection: {e}")
        return jsonify({'error': 'Detection failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
