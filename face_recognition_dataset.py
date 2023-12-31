from face_recognition_settings import FR_DATASET_DIR, FR_USER_DATASET_DIR, IMAGE_EXT, RUN_DEVICE
import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms
from PIL import Image


class ImagesDataset(Dataset):

    def __init__(self, images_dataset_dir, dataset_size=100, device=RUN_DEVICE):
        self.dataset_size = dataset_size
        self.device = device
        self.images_dir = os.path.join(os.getcwd(), images_dataset_dir)
        print('Images Dir = ', self.images_dir)

        self.image_files = [f for f in os.listdir(self.images_dir) if os.path.isfile(os.path.join(self.images_dir, f)) and f.endswith(IMAGE_EXT)]
        random.shuffle(self.image_files)
        self.image_files = self.image_files[:self.dataset_size]
        print('Num Image Files taken = ', len(self.image_files))

        self.img_transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        img_tensor = self.apply_image_transforms(Image.open(image_path)).to(self.device)
        return img_tensor

    def apply_image_transforms(self, image):
        return self.img_transforms(image)


class FaceRecognitionSiameseDataset(Dataset):

    def __init__(self, dataset_size=3500, device=RUN_DEVICE):
        self.device = device

        # image dirs
        self.negative_images_dir = os.path.join(os.getcwd(), FR_DATASET_DIR)
        self.positive_images_dir = os.path.join(os.getcwd(), FR_USER_DATASET_DIR)
        print('Negative Images Dir = ', self.negative_images_dir)
        print('Positive Images Dir = ', self.positive_images_dir)

        # Image files
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

        # Ground truth values
        dataset_flags_true = [True] * int(dataset_size / 2) # 50 percent of dataset will have matching images (user images)
        dataset_flags_false = [False] * int(dataset_size / 2) # 50 percentage of dataset will have differnet images
        self.dataset_flags = dataset_flags_true + dataset_flags_false

        random.shuffle(self.dataset_flags)
        random.shuffle(self.positive_images)
        random.shuffle(self.negative_images)

        self.img_transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        # Picking two images -----

        # Image 1 is a positive image
        img1_idx = random.randint(0, self.num_positive_images-1)
        img1_path = os.path.join(self.positive_images_dir, self.positive_images[img1_idx])

        # Image 2 is positive or negative based on flag
        if self.dataset_flags[idx]:
            # Positive image. img_1 and img_2 are the same class image (user)
            img2_idx = random.randint(0, self.num_positive_images-1)
            img2_path = os.path.join(self.positive_images_dir, self.positive_images[img2_idx])
            y = torch.ones(1, dtype=torch.float32, device=RUN_DEVICE) # 1
        else:
            # Negative image. img_1  and img_2 are different images
            img2_idx = random.randint(0, self.num_negative_images-1)
            img2_path = os.path.join(self.negative_images_dir, self.negative_images[img2_idx])
            y = torch.zeros(1, dtype=torch.float32, device=RUN_DEVICE) # 0

        # Convert to tensor images
        img1_tensor = self.apply_image_transforms(Image.open(img1_path)).to(self.device)
        img2_tensor = self.apply_image_transforms(Image.open(img2_path)).to(self.device)

        return img1_tensor, img2_tensor, y

    def apply_image_transforms(self, image):
        return self.img_transforms(image)

    def shuffle_pos_neg_images(self):
        random.shuffle(self.positive_images)
        random.shuffle(self.negative_images)
