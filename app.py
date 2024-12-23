from flask import Flask, request, render_template, redirect, url_for, session ,jsonify , flash
from pymongo import MongoClient
import cv2
import dlib
import numpy as np
import pickle
import os
import time
import base64
import subprocess
import logging
import threading
import pickle
from PIL import Image
import io
import re






app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret key

logging.basicConfig(level=logging.INFO)




# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition_db']
users_collection = db['users']


TRAINING_SCRIPT_PATH = os.path.join(os.getcwd(), '/home/goodboy/Documents/FinalYearProject-20241002T070942Z-001/FinalYearProject/train_faces.py')

# Load the Dlib face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor_path = "/home/goodboy/Documents/FinalYearProject-20241002T070942Z-001/FinalYearProject/shape_predictor_68_face_landmarks.dat"  # Update with your shape predictor path
recognition_model_path = "/home/goodboy/Documents/FinalYearProject-20241002T070942Z-001/FinalYearProject/dlib_face_recognition_resnet_model_v1.dat"  # Update with your recognition model path

# Load the shape predictor and face recognition model
predictor = dlib.shape_predictor(predictor_path)
encoder = dlib.face_recognition_model_v1(recognition_model_path)


def run_training_script(username):
    """
    Function to run the training script asynchronously.
    """
    try:
        logging.info(f"Starting training for user: {username}")
        # Execute the training script with the username as an argument
        result = subprocess.run(['python3', TRAINING_SCRIPT_PATH, username],
                                capture_output=True, text=True)

        if result.returncode == 0:
            logging.info(f"Training completed successfully for user: {username}")
            logging.info(f"Training Output:\n{result.stdout}")
        else:
            logging.error(f"Training failed for user: {username}")
            logging.error(f"Training Error:\n{result.stderr}")
    except Exception as e:
        logging.error(f"Exception occurred while running training script for user {username}: {e}")



def compute_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def recognize_faces(input_image_path):
    # Load the face encodings and histograms from the file
    try:
        with open('face_encodings.pkl', 'rb') as f:
            face_encodings = pickle.load(f)

        with open('face_histograms.pkl', 'rb') as f:
            face_histograms = pickle.load(f)

        print("Face encodings and histograms loaded successfully.")
    except FileNotFoundError:
        print("No saved encodings or histograms found. Please run the training code first.")
        return

    # Read the input image
    image = cv2.imread(input_image_path)
    if image is None:
        print("Image not found.")
        return

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image using Dlib
    detections = detector(image_rgb, 1)

    # List to store new face encodings and histograms
    face_encodings_new = []
    face_histograms_new = []

    for detection in detections:
        # Get the landmarks for the detected face
        landmarks = predictor(image_rgb, detection)

        # Get the face encoding
        face_encoding = np.array(encoder.compute_face_descriptor(image_rgb, landmarks))
        face_encodings_new.append(face_encoding)

        # Compute color histogram
        hist = compute_histogram(image_rgb[detection.top():detection.bottom(), detection.left():detection.right()])
        face_histograms_new.append(hist)

    # Compare the new encodings with the stored encodings
    for face_encoding, hist_new in zip(face_encodings_new, face_histograms_new):
        matches = []
        for user, (encodings, hists) in zip(face_encodings.keys(), zip(face_encodings.values(), face_histograms.values())):
            # Calculate distances to find the closest match
            distances = np.linalg.norm(encodings - face_encoding, axis=1)
            color_similarities = [cv2.compareHist(hist_new, h, cv2.HISTCMP_CORREL) for h in hists]

            # Combine metrics (you can adjust the weights as needed)
            combined_scores = [0.5 * dist + 0.5 * (1 - sim) for dist, sim in zip(distances, color_similarities)]

            matches.append((user, np.min(combined_scores)))  # (user, combined score)

        # Find the best match
        best_match = min(matches, key=lambda x: x[1])  # Get the user with the smallest distance

        # Define a threshold for matching
        threshold = 0.6  # This value may need tuning based on your dataset
        if best_match[1] < threshold:
            return best_match[0]
            
        else:
            return "naveen"


@app.route('/')
def home():
    username = session.get('username')
    if 'username' in session:  # Check if user is logged in
        return render_template('index_login.html',username = username)
    else:
        return render_template('index.html')

@app.route('/login')  # Added leading slash
def login_page():
    return render_template('login.html')

@app.route('/register')  # Added leading slash
def register_page():
    return render_template('register.html')



@app.route('/train_model', methods=['GET'])
def training():
    try:
        # Run the external Python script using subprocess
        # It's better to use the full path to ensure the script is found
        script_path = os.path.join(os.getcwd(), 'train_model.py')
        
        if not os.path.isfile(script_path):
            return jsonify({"status": "error", "message": "Training script not found"}), 404

        logging.info("Starting model training...")

        # Run the training script
        result = subprocess.run(['python3', script_path], capture_output=True, text=True)

        # Check if the script was executed successfully
        if result.returncode == 0:
            logging.info("Model training completed successfully.")
            return jsonify({"status": "success", "output": result.stdout}), 200
        else:
            logging.error(f"Training script failed: {result.stderr}")
            return jsonify({"status": "error", "output": result.stderr}), 500

    except Exception as e:
        logging.error(f"Error running training script: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/capture_photos/<username>')
def capture_photos(username):
    return render_template('capture_photos.html', username=username)

@app.route('/registering', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if user already exists
        if users_collection.find_one({'username': username}):
            return "User already exists."
        
        # Save user credentials to MongoDB
        users_collection.insert_one({'username': username, 'password': password})
        
        # Create a directory for the user to store photos, named after the username
        user_folder = f'./static/photos/{username}'  # Define where to save photos
        os.makedirs(user_folder, exist_ok=True)
        
        # Redirect to the photo capture page
        return redirect(url_for('capture_photos', username=username))
    
    return render_template('register.html')

@app.route('/save_photo', methods=['POST'])
def save_photo():
    data = request.get_json()
    
    # Check for expected data
    if not data or 'username' not in data or 'imgData' not in data:
        return jsonify({"error": "Missing username or image data"}), 400
    
    username = data['username']
    img_data = data['imgData']

    # Decode the base64 string and prepare for saving
    try:
        img_data = img_data.split(',')[1]  # Remove the header
    except IndexError:
        return jsonify({"error": "Invalid image data format"}), 400

    try:
        img_data = base64.b64decode(img_data)  # Decode the base64 string
    except base64.binascii.Error:
        return jsonify({"error": "Base64 decoding failed"}), 400

    # Define the user directory
    user_dir = os.path.join('/home/goodboy/Documents/FinalYearProject-20241002T070942Z-001/FinalYearProject/Celebrity_Faces_Dataset', username)
    os.makedirs(user_dir, exist_ok=True)

    # Determine the next photo number
    try:
        photo_count = len([name for name in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, name))]) + 1
    except FileNotFoundError:
        return jsonify({"error": "User directory not found and could not be created"}), 400

    photo_path = os.path.join(user_dir, f'photo_{photo_count}.jpg')  # Save as JPEG for quality

    # Convert the byte data to a NumPy array and decode it using OpenCV
    try:
        img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # Decode to an OpenCV image

        if img is None:
            return jsonify({"error": "Image decoding failed"}), 400

        # Save the image with specified quality (1-100, where 100 is the highest quality)
        quality = 95  # Adjust as necessary
        cv2.imwrite(photo_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])  # Set quality for JPEG
    except Exception as e:
        logging.error(f"Error saving photo for user {username}: {e}")
        return jsonify({"error": f"Error saving photo: {e}"}), 400

    logging.info(f"Saved photo {photo_count} for user {username} at {photo_path}")

    # After saving the photo, check if we've reached 25 photos
    try:
        total_photos = len([name for name in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, name))])
    except Exception as e:
        logging.error(f"Error counting photos for user {username}: {e}")
        return jsonify({"error": f"Error counting photos: {e}"}), 500

    # If 25 photos have been saved, trigger the training script
    if total_photos == 33:
        logging.info(f"User {username} has reached 25 photos. Triggering training script.")
        # Run the training script in a separate thread to avoid blocking
        training_thread = threading.Thread(target=run_training_script, args=(username,))
        training_thread.start()

    return jsonify({"status": "success", "photo_path": photo_path, "total_photos": total_photos}), 200





@app.route('/logging', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users_collection.find_one({'username': username, 'password': password})
        if user:
            session['username'] = username
            return redirect(url_for('face_recognition'))
        else:
            return "Invalid username or password."
    
    return render_template('login.html')

@app.route('/face_recognition')
def face_recognition():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Capture video and recognize faces
    # You can use OpenCV to capture video and pass frames to the recognize_face function
    return render_template('face_recognition.html', username='{{ username }}')


@app.route('/recognition', methods=['POST'])
def recognition():
    data = request.json
    username = data.get('username')
    img_data = data.get('imgData')

    # Extract the Base64 string (remove the header)
    img_data = re.sub('^data:image/png;base64,', '', img_data)

    # Decode the Base64 string
    img_bytes = base64.b64decode(img_data)

    # Convert bytes to a PIL Image
    image = Image.open(io.BytesIO(img_bytes))

    # Ensure the images directory exists
    os.makedirs('images', exist_ok=True)

    # Save the image as JPEG
    file_path = os.path.join('images', f'{username}_captured_image.jpg')
    image.convert("RGB").save(file_path, 'JPEG')

    # Call the recognize_faces function with the saved image path
    result = recognize_faces(file_path)
    print(f"Recognized result: {result}")  # Log recognized result

    # Check if the recognized username matches the session username
    session_username = session.get('username')
    print(f"Session username: {session_username}")  # Log session username

    if session_username == result:
        return jsonify(message="Redirecting to home."), 200  # Successful recognition
    else:
        return jsonify(message="Recognition failed. Logging out.")
   







@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
