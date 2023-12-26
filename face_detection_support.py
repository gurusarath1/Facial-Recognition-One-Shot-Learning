import cv2
from face_detection_settings import FACE_CASCADE_FILE
import copy

face_cascade = None


def init_face_detection():
    global face_cascade

    # Load the face cascade file
    # https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_FILE)


def run_face_detection(cv2_img):
    assert face_cascade is not None

    image = copy.deepcopy(cv2_img)

    # Convert to GrayScale (Haar Cascade algorithm works on gray scale images)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Run the face detection algorithm
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    image_with_boxes = None
    # Draw bounding boxes
    for (x, y, w, h) in faces:
        image_with_boxes = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

    return faces, image_with_boxes
