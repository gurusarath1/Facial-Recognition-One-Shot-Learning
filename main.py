import cv2
import os
import uuid
from face_detection_settings import FRAME_DEFAULT_TEXT_COLOR, FRAME_WARNING_TEXT_COLOR, FRAME_ALERT_TEXT_COLOR, \
    FRAME_DEFAULT_TEXT_FONT, FACE_CASCADE_FILE
from face_detection_support import init_face_detection, run_face_detection
from face_recognition_settings import RUN_MODE, COLLECT_FACE_IMAGES_DATA, VERIFY_FACE, USER_FACE_IMAGES_DIR, IMAGE_EXT, \
    PREPROCESS_TRAIN_IMAGES, RUN_AUGMENTATION, TRAIN_LOOP
from face_recognition_support import extract_1face_image, extract_1face_and_preprocess, process_train_images
from ml_utils import run_image_augmentation
from face_recognition_dataset import FaceRecognitionSiameseDataset

if __name__ == '__main__':

    if RUN_MODE == RUN_AUGMENTATION:
        print('RUN_AUGMENTATION')
        run_image_augmentation('./user_images',out_images_dir='user_augmented_images')
        exit()

    if RUN_MODE == TRAIN_LOOP:
        print('Training Siamese Network .. .. ..')
        dataset = FaceRecognitionSiameseDataset()
        print(dataset[0])
        exit()


    # initialize face cascade object
    init_face_detection()

    if RUN_MODE == PREPROCESS_TRAIN_IMAGES:
        process_train_images()
        exit()

    # START THE WEB CAM
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        check, frame = cap.read()

        faces, frame_processed = run_face_detection(frame)

        if len(faces) == 0:
            print('No Face Detected !!')
            cv2.putText(frame_processed, 'No Face', (10, 450), FRAME_DEFAULT_TEXT_FONT, 3, FRAME_ALERT_TEXT_COLOR, 2,
                        cv2.LINE_AA)
            continue
        elif len(faces) > 1:
            print('More than one face detected !')
            cv2.putText(frame_processed, str(len(faces)), (10, 450), FRAME_DEFAULT_TEXT_FONT, 3,
                        FRAME_WARNING_TEXT_COLOR, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_processed, '1', (10, 450), FRAME_DEFAULT_TEXT_FONT, 3, FRAME_DEFAULT_TEXT_COLOR, 2,
                        cv2.LINE_AA)

        image_face_cut = frame # This line is to ensure image_face_cut is always defined
        if RUN_MODE == COLLECT_FACE_IMAGES_DATA:
            image_face_cut = extract_1face_and_preprocess(frame_processed, faces[0])
            image_file_name = str(uuid.uuid1()) + IMAGE_EXT
            file_path = str(os.path.join(os.getcwd(), USER_FACE_IMAGES_DIR, image_file_name))
            print(file_path)
            cv2.imwrite(file_path, image_face_cut)
        elif RUN_MODE == VERIFY_FACE:
            pass
        else:
            assert False

        # Show frame
        cv2.imshow('video stream', image_face_cut)

        key = cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
