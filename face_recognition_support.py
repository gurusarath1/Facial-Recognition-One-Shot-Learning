import numpy as np
import cv2
import os
import torch
from face_recognition_settings import TRAINING_IMAGE_SIZE, DATASET_DIR, PROCESSED_TRAIN_DATA_DIR, IMAGE_EXT, RUN_DEVICE, \
    BATCH_SIZE, NUM_EPOCHS, MODEL_SAVE_PATH, CONTRASTIVE_LOSS_TRAIN_MARGIN, CONTRASTIVE_LOSS_EVAL_MIN_DIST, \
    NUM_TEST_IMAGES_TO_VERIFY, EVAL_RATIO_SUCCESS_VOTES
from face_detection_support import run_face_detection
from torch.utils.data import DataLoader
from siamese_network_model import cnn_80_encoder, siamese_network
from face_recognition_dataset import FaceRecognitionSiameseDataset
from datetime import date
from ml_utils import save_torch_model, load_torch_model
from torchvision.transforms import transforms
import matplotlib.pyplot as plt


def extract_1face_and_preprocess(image: np.ndarray, face_bounding_box: tuple, ouput_size: tuple = None) -> np.ndarray:
    image_size = TRAINING_IMAGE_SIZE
    if ouput_size:
        image_size = ouput_size

    assert (face_bounding_box[2] >= image_size[0] and face_bounding_box[2] >= image_size[1])

    face_cut = extract_1face_image(image, face_bounding_box) # get only the face
    train_size_face = cv2.resize(face_cut, image_size, interpolation=cv2.INTER_AREA) # resize the image to training size
    return train_size_face


def extract_1face_image(image: np.ndarray, face_bounding_box: tuple) -> np.ndarray:
    (x, y, w, h) = face_bounding_box
    face_cut = image[y:y + w, x:x + h, :]
    return face_cut


def process_train_images():
    for dir in os.listdir(DATASET_DIR):
        for face_file in os.listdir(os.path.join(DATASET_DIR, dir)):
            full_image_path = os.path.join(DATASET_DIR, dir, face_file)
            print('Processing file -- ', full_image_path)

            image = cv2.imread(full_image_path)
            faces, box_image = run_face_detection(image) #

            # Skip files with more than 2 faces or no faces
            if len(faces) > 1 or len(faces) == 0:
                print('Skip file -- ', full_image_path)
                continue

            w = faces[0][2]
            h = faces[0][3]

            if w < TRAINING_IMAGE_SIZE[0] and h < TRAINING_IMAGE_SIZE[1]:
                print('Skip file (small) -- ', full_image_path)
                continue

            # Get only the face
            train_image = extract_1face_and_preprocess(image, faces[0])

            output_file_name = face_file.split('.')[0] + '_face' + IMAGE_EXT
            output_file_path = os.path.join(os.getcwd(), PROCESSED_TRAIN_DATA_DIR, output_file_name)
            print('Ouput file -- ', output_file_path)

            cv2.imwrite(output_file_path, train_image)


# Distance between image encodings
def l2_dist(x1, x2):
    l2_dist = torch.unsqueeze(torch.sum(torch.pow(x1 - x2, 2), dim=1), dim=1)
    return l2_dist


def contrastive_loss(x1, x2, y, m=CONTRASTIVE_LOSS_TRAIN_MARGIN):
    l2_dist = torch.unsqueeze(torch.sum(torch.pow(x1 - x2, 2), dim=1), dim=1)
    m = torch.Tensor(np.ones_like(y.cpu().numpy()) * m).to(RUN_DEVICE)
    g = (m - l2_dist)
    g2 = torch.cat([torch.zeros_like(y), g], dim=1)
    g3 = torch.unsqueeze(torch.max(g2, dim=1).values, dim=1)
    L = torch.mean(((1 - y) * l2_dist) + (y * g3))
    return L, l2_dist


def train_loop():

    # Dataset
    dataset = FaceRecognitionSiameseDataset()
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # DL Model
    enc = cnn_80_encoder()
    siamese_net = siamese_network(model=enc).to(RUN_DEVICE)

    # Optimizer
    optimz = torch.optim.Adam(siamese_net.parameters(), lr=0.01, weight_decay=1e-2)

    # Get the last check point model weights
    load_torch_model(siamese_net, file_name='siamese_net', path=MODEL_SAVE_PATH, load_latest=True)

    # Set the model in train mode
    siamese_net.train()

    # History for plotting graph
    acc_history = []
    loss_history = []
    for epoch in range(NUM_EPOCHS):

        dataset.shuffle_pos_neg_images()

        for batch_idx, batch in enumerate(train_dataloader):
            # Get the two sets of image batches and y
            input_images_1 = batch[0]
            input_images_2 = batch[1]
            ground_truth = batch[2]

            # Run the images through the model
            enc1, enc2 = siamese_net(input_images_1, input_images_2)
            pred_loss, l2_dist = contrastive_loss(enc1, enc2, ground_truth)

            # Backpropagation
            pred_loss.backward()
            optimz.step()
            optimz.zero_grad()
            loss_history.append(pred_loss.item())

            # Checkpoint
            if batch_idx == 0:
                pred_classes = (torch.squeeze(l2_dist) > CONTRASTIVE_LOSS_EVAL_MIN_DIST)
                comp = (pred_classes == torch.squeeze(ground_truth)).to(torch.float32)
                acc = torch.mean(comp)
                acc_history.append(acc.item())
                print(f'epoch = {epoch}  batch={batch_idx}  batch_shape={batch[0].shape}  acc={acc.item()}')
                print(f'Loss = {pred_loss.item()}')
                file_name_info = '_' + str(date.today()) + '_' + str(pred_loss.item())
                save_torch_model(siamese_net, file_name='siamese_net',
                                 additional_info=file_name_info,
                                 path=MODEL_SAVE_PATH,
                                 two_copies=True)

    # Simple Plotting at the end of train loop
    plt.plot(loss_history)
    plt.show()
    plt.plot(acc_history)
    plt.show()


def mission_mode(in_rgb_numpy_image, siamese_net, user_reference_images):
    num_test_images_to_verify = NUM_TEST_IMAGES_TO_VERIFY

    # Cut Face Image from to verify (from webcam)
    in_img = torch.unsqueeze(transforms.ToTensor()(in_rgb_numpy_image), dim=0).to(RUN_DEVICE)

    # Get prediction of input image against all(NUM_TEST_IMAGES_TO_VERIFY) images in the user image dataset
    num_votes = 0
    for i in range(num_test_images_to_verify):
        user_image = torch.unsqueeze(user_reference_images[i], dim=0)
        enc1, enc2 = siamese_net(user_image, in_img)
        vec_dist = l2_dist(enc1, enc2)  # similarity metric
        pred = vec_dist
        pred = torch.squeeze(pred).item()
        print('Distance from user image = ', pred)

        # If the images are similar, increase the vote count
        if pred < CONTRASTIVE_LOSS_EVAL_MIN_DIST:
            num_votes += 1

    print(f'num_votes = {num_votes}')
    confidence = (num_votes / num_test_images_to_verify)

    # Percentage of votes to win
    if confidence > EVAL_RATIO_SUCCESS_VOTES:
        print('VERDICT - USER VERIFIED !!!')
    else:
        print('VERDICT - UNKNOWN PERSON')
