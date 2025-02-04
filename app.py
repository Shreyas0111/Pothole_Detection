from flask import Flask, render_template, request, jsonify  # type: ignore
from ultralytics import YOLO  # type: ignore
import cv2
import numpy as np
import base64
import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy  # type: ignore
from geopy.distance import geodesic  # type: ignore

app = Flask(__name__)

# Configure PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:sn0111@localhost/pothole_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define folder for saving images
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("/mnt/c/Users/Shreyas N/Desktop/PDaM/models/best_saved_model/best_float16.tflite")


# Define Pothole model
class Pothole(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    image_path = db.Column(db.String(255), nullable=False)
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)


# Create DB tables if they don't exist
with app.app_context():
    db.create_all()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/map', methods=['GET'])
def map_view():
    return render_template('map.html')



@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        img_array = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        results = model.predict(frame)

        # Check if potholes were detected
        if results and len(results) > 0 and len(results[0].boxes) > 0:
            annotated_frame = results[0].plot()

            # Generate unique image filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            image_filename = f"pothole_{timestamp}.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

            # Save the image
            cv2.imwrite(image_path, annotated_frame)

            # Sample Latitude & Longitude (Replace with actual GPS data)
            latitude = 13.032600  # Example lat (Bangalore)
            longitude = 77.592845  # Example long (Bangalore)

            # Save pothole info to DB **only if a pothole was detected**
            new_pothole = Pothole(
                latitude=latitude,
                longitude=longitude,
                image_path=image_path
            )
            db.session.add(new_pothole)
            db.session.commit()

            print("Pothole detected and saved.")

        else:
            print("No potholes detected. Skipping database update.")
            annotated_frame = frame  # Return original frame without saving

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        processed_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'processed_image': f'data:image/jpeg;base64,{processed_img}'})

    except Exception as e:
        print(f"Error in detection: {e}")
        db.session.rollback()  # Rollback in case of error
        return jsonify({'error': 'Detection failed'}), 500



@app.route('/nearby_potholes', methods=['GET'])
def nearby_potholes():
    try:
        latitude = float(request.args.get('latitude'))
        longitude = float(request.args.get('longitude'))
        radius = float(request.args.get('radius', 1000))  # Default 1 km radius

        potholes = Pothole.query.all()
        nearby_potholes = []

        for pothole in potholes:
            pothole_location = (pothole.latitude, pothole.longitude)
            user_location = (latitude, longitude)

            # Calculate distance between user and pothole
            distance = geodesic(user_location, pothole_location).meters
            if distance <= radius:
                nearby_potholes.append({
                    'latitude': pothole.latitude,
                    'longitude': pothole.longitude,
                    'image_path': pothole.image_path
                })

        return jsonify(nearby_potholes)

    except Exception as e:
        print(f"Error in fetching nearby potholes: {e}")
        return jsonify({'error': 'Failed to fetch potholes'}), 500


if __name__ == '__main__':
    app.run(debug=True)
