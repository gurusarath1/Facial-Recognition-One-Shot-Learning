from face_recognition_settings import FR_DATASET_DIR, FR_USER_DATASET_DIR, IMAGE_EXT, RUN_DEVICE
import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImagesDataset(Dataset):

    def __init__(self, images_dataset_dir, dataset_size=1000, device=RUN_DEVICE):
        self.dataset_size = dataset_size
        self.device = device
        self.images_dir = os.path.join(os.getcwd(), images_dataset_dir)
        print('Images Dir = ', self.images_dir)

        self.image_files = [f for f in os.listdir(self.images_dir) if os.path.isfile(os.path.join(self.images_dir, f)) and f.endswith(IMAGE_EXT)]
        random.shuffle(self.image_files)
        self.image_files = self.image_files[:self.dataset_size]

        print('Num Image Files taken = ', len(self.image_files))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        img_tensor = read_image(image_path).to(self.device)
        return img_tensor


class FaceRecognitionSiameseDataset(Dataset):

    def __init__(self, dataset_size=1000, device=RUN_DEVICE):
        self.device = RUN_DEVICE
        self.negative_images_dir = os.path.join(os.getcwd(), FR_DATASET_DIR)
        self.positive_images_dir = os.path.join(os.getcwd(), FR_USER_DATASET_DIR)
        print('Negative Images Dir = ', self.negative_images_dir)
        print('Positive Images Dir = ', self.positive_images_dir)

        self.negative_images = [f for f in os.listdir(self.negative_images_dir) if
                                os.path.isfile(os.path.join(self.negative_images_dir, f)) and f.endswith(IMAGE_EXT)]
        self.positive_images = [f for f in os.listdir(self.positive_images_dir) if
                                os.path.isfile(os.path.join(self.positive_images_dir, f)) and f.endswith(IMAGE_EXT)]
        self.num_negative_images = len(self.negative_images)
        self.num_positive_images = len(self.positive_images)
        self.dataset_size = dataset_size
        print('len negative_images = ', self.num_negative_images)
        print('len positive_images = ', self.num_positive_images)
        print('dataset_size = ', self.dataset_size)

        dataset_flags_true = [True] * int(dataset_size / 2)
        dataset_flags_false = [False] * int(dataset_size / 2)
        self.dataset_flags = dataset_flags_true + dataset_flags_false
        random.shuffle(self.dataset_flags)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        # Image 1 is a positive image
        img1_idx = random.randint(0, self.num_positive_images)
        img1_path = os.path.join(self.positive_images_dir, self.positive_images[img1_idx])

        # Image 2 is positive or negative based on flag
        if self.dataset_flags[idx]:
            img2_idx = random.randint(0, self.num_positive_images)
            img2_path = os.path.join(self.positive_images_dir, self.positive_images[img2_idx])
            y = 1
        else:
            img2_idx = random.randint(0, self.num_negative_images)
            img2_path = os.path.join(self.negative_images_dir, self.negative_images[img2_idx])
            y = 0

        img1_tensor = read_image(img1_path).to(self.device)
        img2_tensor = read_image(img2_path).to(self.device)

        return img1_tensor, img2_tensor, y
