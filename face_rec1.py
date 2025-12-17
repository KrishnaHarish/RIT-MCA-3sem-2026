import cv2
import face_recognition
import os
import numpy as np

# Load known images and generate encodings
known_faces_encodings = []
known_faces_names = []

# Directory containing known images
image_dir = "images" 
for filename in os.listdir(image_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        # Load image file
        image_path = os.path.join(image_dir, filename)
        image = face_recognition.load_image_file(image_path)
        
        # Get face encoding (a 128-dimension vector)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_faces_encodings.append(encoding[0])
            # Use the filename (without extension) as the name
            known_faces_names.append(os.path.splitext(filename)[0].capitalize())

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the webcam frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face with known faces
        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)

        name = "Unknown"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Press 'q' on the keyboard to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
