import cv2
from face_detection_settings import FRAME_DEFAULT_TEXT_COLOR, FRAME_WARNING_TEXT_COLOR, FRAME_ALERT_TEXT_COLOR, FRAME_DEFAULT_TEXT_FONT, FACE_CASCADE_FILE
from face_recognition_support import extract_1face_image

if __name__ == '__main__':

    # Load the face and eye cascade files
    # https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_FILE)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        check, frame = cap.read()

        # Make a copy of the frame
        frame_processed = frame.copy()
        # Convert to GrayScale (Haar Cascade algorithm works on gray scale images)
        gray_frame = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)

        # Run the face detection algorithm
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        if len(faces) == 0:
            print('No Face Detected !!')
            cv2.putText(frame_processed, 'No Face', (10, 450), FRAME_DEFAULT_TEXT_FONT, 3, FRAME_ALERT_TEXT_COLOR, 2, cv2.LINE_AA)
            continue
        elif len(faces) > 1:
            print('More than one face detected !')
            cv2.putText(frame_processed, str(len(faces)), (10, 450), FRAME_DEFAULT_TEXT_FONT, 3, FRAME_WARNING_TEXT_COLOR, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_processed, '1', (10, 450), FRAME_DEFAULT_TEXT_FONT, 3, FRAME_DEFAULT_TEXT_COLOR, 2, cv2.LINE_AA)

        # Draw bounding boxes
        for (x, y, w, h) in faces:
            print(w,h)
            frame_processed = cv2.rectangle(frame_processed, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray_frame[y:y + h, x:x + w]
            roi_color = frame_processed[y:y + h, x:x + w]

            # Process only one face (random)
            break

        image_face_box = extract_1face_image(frame_processed, faces[0])

        # Show frame
        cv2.imshow('video stream', image_face_box)

        key = cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()