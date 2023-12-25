import cv2

# https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
FACE_CASCADE_FILE = 'haarcascade_frontalface_default.xml'
FRAME_DEFAULT_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

# colors in BGR format
FRAME_DEFAULT_TEXT_COLOR = (0, 255, 0)
FRAME_ALERT_TEXT_COLOR = (0, 0, 255)
FRAME_WARNING_TEXT_COLOR = (0, 255, 255)
