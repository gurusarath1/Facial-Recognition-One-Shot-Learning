import numpy as np
import cv2
import os
from face_recognition_settings import TRAINING_IMAGE_SIZE, DATASET_DIR, PROCESSED_TRAIN_DATA_DIR, IMAGE_EXT
from face_detection_support import init_face_detection, run_face_detection
import uuid

def extract_1face_and_preprocess(image: np.ndarray, face_bounding_box: tuple, ouput_size: tuple = None) -> np.ndarray:
    image_size = TRAINING_IMAGE_SIZE
    if ouput_size:
        image_size = ouput_size

    assert (face_bounding_box[2] >= image_size[0] and face_bounding_box[2] >= image_size[1])

    face_cut = extract_1face_image(image, face_bounding_box)
    train_size_face = cv2.resize(face_cut, image_size, interpolation=cv2.INTER_AREA)
    return train_size_face

def extract_1face_image(image: np.ndarray, face_bounding_box: tuple) -> np.ndarray:
    (x, y, w, h) = face_bounding_box
    face_cut = image[y:y+w, x:x+h, :]
    return face_cut


def process_train_images():

    for dir in os.listdir(DATASET_DIR):
        for face_file in os.listdir(os.path.join(DATASET_DIR,dir)):
            full_image_path = os.path.join(DATASET_DIR, dir, face_file)
            print('Processing file -- ', full_image_path)

            image = cv2.imread(full_image_path)
            faces, box_image = run_face_detection(image)

            if len(faces) > 1 or len(faces) == 0:
                print('Skip file -- ', full_image_path)
                continue

            w = faces[0][2]
            h = faces[0][3]

            if w < TRAINING_IMAGE_SIZE[0] and h < TRAINING_IMAGE_SIZE[1]:
                print('Skip file (small) -- ', full_image_path)
                continue


            train_image = extract_1face_and_preprocess(image, faces[0])

            output_file_name = face_file.split('.')[0] + '_face' + IMAGE_EXT
            output_file_path = os.path.join(os.getcwd(), PROCESSED_TRAIN_DATA_DIR, output_file_name)
            print('Ouput file -- ', output_file_path)

            cv2.imwrite(output_file_path, train_image)