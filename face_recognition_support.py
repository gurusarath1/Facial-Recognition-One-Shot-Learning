import numpy as np
import cv2
import os
import torch
from face_recognition_settings import TRAINING_IMAGE_SIZE, DATASET_DIR, PROCESSED_TRAIN_DATA_DIR, IMAGE_EXT, RUN_DEVICE, BATCH_SIZE, NUM_EPOCHS, MODEL_SAVE_PATH, USER_FACE_IMAGES_DIR
from face_detection_support import init_face_detection, run_face_detection
from torch.utils.data import DataLoader
from siamese_network_model import cnn_80_encoder, siamese_network
from face_recognition_dataset import FaceRecognitionSiameseDataset, ImagesDataset
from datetime import date
from ml_utils import save_torch_model, load_torch_model
from torchvision.transforms import transforms
from PIL import Image

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

def train_loop():
    dataset = FaceRecognitionSiameseDataset()
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    enc = cnn_80_encoder()
    siamese_net = siamese_network(model=enc).to(RUN_DEVICE)

    loss_fn = torch.nn.BCELoss()
    optimz = torch.optim.Adam(siamese_net.parameters(), lr=0.001)

    load_torch_model(siamese_net, file_name='siamese_net', path=MODEL_SAVE_PATH, load_latest=True)

    siamese_net.train()
    for epoch in range(NUM_EPOCHS):

        dataset.shuffle_pos_neg_images()

        for batch_idx, batch in enumerate(train_dataloader):
            input_images_1 = batch[0]
            input_images_2 = batch[1]
            ground_truth = batch[2]

            pred = siamese_net(input_images_1, input_images_2)
            pred_loss = loss_fn(pred, ground_truth)

            # Backpropagation
            pred_loss.backward()
            optimz.step()
            optimz.zero_grad()

            if batch_idx == 0:
                pred_classes = (torch.squeeze(pred) > 0.5)
                comp = (pred_classes == torch.squeeze(ground_truth)).to(torch.float32)
                acc = torch.mean(comp)
                print(f'epoch = {epoch}  batch={batch_idx}  batch_shape={batch[0].shape}  acc={acc.item()}')
                print(f'Loss = {pred_loss.item()}')
                file_name_info = '_' + str(date.today()) + '_' + str(pred_loss.item())
                save_torch_model(siamese_net, file_name='siamese_net',
                                 additional_info=file_name_info,
                                 path=MODEL_SAVE_PATH,
                                 two_copies=True)

def mission_mode(in_rgb_numpy_image, siamese_net):

    num_test_images_to_verify = 5
    in_img = torch.unsqueeze(transforms.ToTensor()(in_rgb_numpy_image), dim=0).to(RUN_DEVICE)


    #user_reference_images = ImagesDataset(images_dataset_dir=USER_FACE_IMAGES_DIR, dataset_size=num_test_images_to_verify)

    ref_img = torch.unsqueeze(transforms.ToTensor()(Image.open('front.jpg')), dim=0).to(RUN_DEVICE)

    num_votes = 0
    for i in range(num_test_images_to_verify):
        #user_image = torch.unsqueeze(user_reference_images[i], dim=0)
        user_image = ref_img
        pred = siamese_net(user_image, in_img)
        pred = torch.squeeze(pred).item()

        if pred > 0.5:
            num_votes += 1

    print(f'num_votes = {num_votes}')
    confidence = (num_votes / num_test_images_to_verify)

    if confidence > 0.7:
        print('This is Guru !!!!!!!!!!!!!!')
    else:
        print(':(')


