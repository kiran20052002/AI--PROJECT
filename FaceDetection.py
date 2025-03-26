import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


image_counter = 0
maximum_images = 3
output_folder = "captured_faces"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # we will Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # we will create rectanle on face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #showing some messages
    cv2.putText(frame, "Press 'k' to capture an image.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to exit.", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('k'):
        if image_count < maximum_images:
            img_name = f"{output_folder}/face_{image_count}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"Image {image_count + 1} saved as {img_name}")
            image_count += 1
        else:
            print("Maximum images captured (3). Press 'q' to exit.")

    
    elif key == ord('q'):
        print("Exiting...")
        break


cap.release()
cv2.destroyAllWindows()