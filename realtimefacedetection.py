import cv2
import numpy as np
from tensorflow.keras.models import model_from_json, Sequential

# Load JSON model
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

# Register Sequential explicitly to avoid loading error
model = model_from_json(model_json, custom_objects={"Sequential": Sequential})
model.load_weights("facialemotionmodel.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the face image
def extract_features(image):
    feature = np.array(image, dtype="float32") / 255.0  # Normalize
    feature = feature.reshape(1, 48, 48, 1)  # Reshape for model
    return feature

# Emotion labels
labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Open webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]  # Extract face
        face_img = cv2.resize(face_img, (48, 48))  # Resize to match model input
        img = extract_features(face_img)  # Preprocess image
        
        # Predict emotion
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        
        # Draw rectangle and put text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, prediction_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Facial Emotion Recognition", frame)

    # Exit when 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
