import cv2
from djitellopy import Tello

# Connecting to Tello
tello = Tello()
tello.connect()
tello.streamon()
cap = tello.get_frame_read()
print(f"Battery Life Percentage: {tello.get_battery()}%")

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 0, 255), 5)
    return faces

while True:
    frame = cap.frame
    frame = cv2.resize(frame, (960, 720))

    faces = detect_bounding_box(
        frame
    ) 

    cv2.imshow("Tello Stream", frame)

    # Press q to exit video feed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

tello.streamoff()
cv2.destroyAllWindows()