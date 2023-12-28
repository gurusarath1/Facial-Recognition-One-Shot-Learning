import torch
import re
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision
import os
import glob
from torch.utils.data import Dataset
import PIL
from tqdm import tqdm
import random
import cv2

SAVE_IMAGE_DIR = './saved_images/'
UTIL_PRINTS = False

'''
Channel Ordering and Shape of image ---

cv2.imread() - BGR (h,w,c)
mpimg.imread() - RGB ()
matplotlib.pyplot.imread - RGB (h,w,c)
torchvision.io.read_image - RGB (c,h,w)
PIL.Image.open - 
'''

#
#
#    OPEN CV functions
#
augmentations_list = ['vertical_flip',
                      'horizontal_flip',
                      'contrast_brightness',
                      'rotation',
                      'horizontal_shift',
                      'vertical_shift',
                      'salt_and_pepper_noise',
                      ]


# Image format = [h,w,c]
def augmentation_functions(img: np.ndarray, augmentation: str = '', *args) -> np.ndarray:
    print(f'augmentation = {augmentation} image_shape = {img.shape}')

    img = img.copy()  # Do not modify the existing image

    if augmentation == 'vertical_flip':  # -----------------------------
        return cv2.flip(img, 0)

    elif augmentation == 'horizontal_flip':  # -----------------------------
        return cv2.flip(img, 1)

    elif augmentation == 'contrast_brightness':  # -----------------------------
        alpha = args[0]  # Contrast Control
        beta = args[1]  # Brightness Control
        print(f'alpha={alpha}  beta={beta}')
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    elif augmentation == 'rotation':  # -----------------------------
        angle = args[0]  # angle in degrees

        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        return img


    elif augmentation == 'horizontal_shift':  # -----------------------------
        dir_ratio = args[0]
        print(f'dir_ratio = {dir_ratio}')
        # Check the range
        assert -1 < dir_ratio < 1

        fill_type = 'nearest'
        if len(args) >= 2:
            fill_type = args[1]

        h, w = img.shape[:2]
        num_pixels_shift = int(w * abs(dir_ratio))

        # RIGHT SHIFT
        if dir_ratio > 0:
            img_cut = img.copy()
            img_cut = img_cut[:, :w - num_pixels_shift, :]

            if fill_type == 'resize':
                return cv2.resize(img_cut, (h, w), cv2.INTER_CUBIC)
            elif fill_type == 'nearest':
                img[:, :, :] = img_cut[:, 0:1, :]  # Pick only the first column and broadcast
                img[:, num_pixels_shift:, :] = img_cut[:, :, :]
                return img

        # LEFT SHIFT
        elif dir_ratio < 0:
            img_cut = img.copy()
            img_cut = img_cut[:, num_pixels_shift:, :]

            if fill_type == 'resize':
                return cv2.resize(img_cut, (h, w), cv2.INTER_CUBIC)
            elif fill_type == 'nearest':
                img[:, :, :] = img_cut[:, -1:, :]  # Pick only the first column and broadcast
                img[:, 0:w - num_pixels_shift, :] = img_cut[:, :, :]
                return img

        else:  # No shift
            return img

    elif augmentation == 'vertical_shift':  # -----------------------------
        dir_ratio = args[0]

        print(f'dir_ratio = {dir_ratio}')

        # Check the range
        assert -1 < dir_ratio < 1

        fill_type = 'nearest'
        if len(args) >= 2:
            fill_type = args[1]

        h, w = img.shape[:2]
        num_pixels_shift = int(h * abs(dir_ratio))

        # DOWN SHIFT
        if dir_ratio > 0:
            img_cut = img.copy()
            img_cut = img_cut[:h - num_pixels_shift, :, :]

            if fill_type == 'resize':
                return cv2.resize(img_cut, (h, w), cv2.INTER_CUBIC)
            elif fill_type == 'nearest':
                img[:, :, :] = img_cut[0:1, :, :]  # Pick only the first column and broadcast
                img[num_pixels_shift:, :, :] = img_cut[:, :, :]
                return img

        # UP SHIFT
        elif dir_ratio < 0:
            img_cut = img.copy()
            img_cut = img_cut[num_pixels_shift:, :, :]

            if fill_type == 'resize':
                return cv2.resize(img_cut, (h, w), cv2.INTER_CUBIC)
            elif fill_type == 'nearest':
                img[:, :, :] = img_cut[-1:, :, :]  # Pick only the first column and broadcast
                img[0:h - num_pixels_shift, :, :] = img_cut[:, :, :]
                return img

        else:  # No shift
            return img

    elif augmentation == 'salt_and_pepper_noise':
        pixel_min = 0
        pixel_max = 255
        p = 0.005  # Probability of salt and pepper noise

        if len(args) >= 3:
            pixel_min = args[0]
            pixel_max = args[1]
            p = args[2]
        elif len(args) == 2:
            pixel_min = args[0]
            pixel_max = args[1]
        elif len(args) == 1:
            pixel_max = args[0]

        p = p / 2  # Probability of only salt/pepper noise

        # Add SALT
        noise_matrix = np.random.uniform(0, 1, size=img.shape[0:2])
        noise_locations_matrix = (noise_matrix < p)
        img[noise_locations_matrix] = pixel_max

        # Add PEPPER
        noise_matrix = np.random.uniform(0, 1, size=img.shape[0:2])
        noise_locations_matrix = (noise_matrix < p)
        img[noise_locations_matrix] = pixel_min

        return img

    else:  # No Augmentation # -----------------------------
        return img


def run_image_augmentation(in_images_path: str, out_images_dir: str = 'augmented_images', image_ext: str = '.jpg'):
    onlyfiles = [f for f in os.listdir(in_images_path) if
                 os.path.isfile(os.path.join(in_images_path, f)) and f.endswith(image_ext)]
    out_images_path = os.path.join(os.getcwd(), out_images_dir)

    for image_file_name in onlyfiles:
        image_path = os.path.join(in_images_path, image_file_name)
        out_image_path = os.path.join(out_images_path, 'aug_' + image_file_name)
        img = cv2.imread(image_path)

        aug_image = img

        if random.choice([True, False]):
            shift_ratio = random.random() * 0.4 - 0.2  # -0.2 to 0.2 ratio
            aug_image = augmentation_functions(aug_image, 'vertical_shift', shift_ratio)
        elif random.choice([True, False]):
            shift_ratio = random.random() * 0.4 - 0.2  # -0.2 to 0.2 ratio
            aug_image = augmentation_functions(aug_image, 'horizontal_shift', shift_ratio)
        elif random.choice([True, False]):
            angle = (random.random() * 90) - 45  # -45 to 45 degree
            aug_image = augmentation_functions(aug_image, 'rotation', angle)

        if random.choice([True, False]):
            aug_image = augmentation_functions(aug_image, 'horizontal_flip')
        elif random.choice([True, False]):
            aug_image = augmentation_functions(aug_image, 'vertical_flip')

        if random.choice([True, False]):
            beta = random.randint(-30, 100)
            aug_image = augmentation_functions(aug_image, 'contrast_brightness', 1, beta)

        if random.choice([True, False]):
            p = random.random() * 0.01
            aug_image = augmentation_functions(aug_image, 'salt_and_pepper_noise', 0, 255, p)

        print(out_image_path)
        cv2.imwrite(out_image_path, aug_image)


#
#
#    TORCH and OTHER functions
#
def get_torch_transforms_pil_image_to_torch_image(new_size=None, minus_1_to_1_scale=False):
    transforms_list = []

    if new_size:
        transforms_list.append(transforms.Resize(new_size))

    transforms_list.append(transforms.ToTensor())  # Scales the pixels 0 - 1 # Changes from PIL/numpy(HxWxC) to (CxHxW)

    if minus_1_to_1_scale:
        transforms_list.append(transforms.Lambda(lambda x: (x * 2) - 1))  # Scale the pixel values to -1 to 1

    return transforms_list


def get_torch_transforms_torch_image_to_pil_image(new_size=None, minus_1_to_1_scaled_image=False):
    transforms_list = []

    if new_size:
        transforms_list.append(transforms.Resize(new_size))

    if minus_1_to_1_scaled_image:
        transforms_list.append(transforms.Lambda(lambda x: (x + 1) / 2))
    else:
        # We assume 0-1 pixel scale in the input
        pass

    transforms_list.append(
        transforms.Lambda(lambda x: x.permute(1, 2, 0)))  # Torch image Batch x Channel x Height x Width --> PIL image
    transforms_list.append(transforms.Lambda(lambda x: x * 255.0))  # 0-1 scale --> 0-255
    transforms_list.append(transforms.Lambda(
        lambda x: x.numpy().astype(np.uint8)))  # torch tensor ->  numpy array; PIL uses datatype of uint8
    transforms_list.append(transforms.ToPILImage())

    return transforms_list


def get_device():
    dev = 'cpu'

    if torch.cuda.is_available():
        dev = 'cuda'

    print(f'Device = {dev}')

    return dev


# Function Source: Book -  Machinelearning with Pytorch and Sklearn - Sbastian .. Pg:514
def string_tokenizer(text: str):
    text = re.sub('<[^>]*>', '', text)

    emoticons = re.findall(
        '(?::|;|=)(?:-)?(?:\)|\(|D|P)',
        text.lower()
    )

    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')

    tokenized = text.split()

    return tokenized


def get_sentences_from_text(text: str):
    pat = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)

    return pat.findall(text.lower())


# Obtained from coursera course on GANs
def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def save_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), file_name='saved_img.png'):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())

    print(f'Saving images to file {file_name}')
    plt.savefig(file_name, bbox_inches='tight')


def get_image_tensor(image_path, device='cpu', transform=None, add_batch_dim=False, batch_dim_index=0):
    image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).to(device)
    # [channels, height, width].

    image = image / 255.0

    if add_batch_dim:
        image = torch.unsqueeze(image, batch_dim_index)
        # [batch_size, channels, height, width]

    if transform is not None:
        image = transform(image)

    print(image_path, '- ', image.shape, device)

    return image


# Reference: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch
def display_image(tensor_image, batch_dim_exist=False, batch_dim_index=0, save_image=False, file_name='saved_img.png'):
    if batch_dim_exist:
        plt.imshow(tensor_image.squeeze(dim=batch_dim_index).permute(1, 2,
                                                                     0))  # remove batch dim and Make the Channel dim last
    else:
        plt.imshow(tensor_image.permute(1, 2, 0))  # Make the Channel dim last

    if save_image:
        plt.savefig(SAVE_IMAGE_DIR + file_name, bbox_inches='tight')
    else:
        plt.show()


def get_numpy_onehot_array(categorical_numpy_array, num_categories=None):
    categorical_numpy_array = categorical_numpy_array.astype(int)

    if not num_categories:
        num_categories = np.max(categorical_numpy_array) + 1

    num_data_points = categorical_numpy_array.shape[0]  # Num samples
    if UTIL_PRINTS: print(f'num_categories = {num_categories}  num_data_points={num_data_points}')

    one_hot_array = np.zeros((num_data_points, num_categories))
    one_hot_array[np.arange(categorical_numpy_array.size), categorical_numpy_array] = 1

    return one_hot_array


def shuffle_two_numpy_array(data_x, data_y):
    num_data_points = data_x.shape[0]  # Num samples
    if UTIL_PRINTS: print(f'num_data_points = {num_data_points}')

    assert num_data_points == data_y.shape[0]

    shuffle_idxs = np.random.permutation(num_data_points)
    return data_x[shuffle_idxs], data_y[shuffle_idxs]


def shuffle_two_torch_array(data_x, data_y):
    num_data_points = data_x.shape[0]  # Num samples
    if UTIL_PRINTS: print(f'num_data_points = {num_data_points}')

    assert num_data_points == data_y.shape[0]

    shuffle_idxs = torch.randperm(num_data_points)
    return data_x[shuffle_idxs], data_y[shuffle_idxs]


def save_torch_model(model, file_name='saved_model', additional_info='', path='./saved_models', two_copies=True):
    torch.save(model.state_dict(), path + '/' + file_name + additional_info)

    if two_copies:
        torch.save(model.state_dict(), path + '/' + file_name + '_LATEST_COPY')


def load_torch_model(model, file_name='saved_model', path='saved_models', load_latest=True):

    if load_latest:
        full_path = os.path.join(path, file_name + '_LATEST_COPY')
    else:
        full_path = os.path.join(path, file_name)

    if os.path.isfile(full_path):
        print(f'Loading Model {full_path} ...')
        model.load_state_dict(torch.load(full_path))
    else:
        print('MODEL FILE NOT FOUND .. .. SKIP LOADING ..')



def get_torch_model_output_size_at_each_layer(model, input_shape=0, input_tensor=None):
    print('=====================================================')
    print(f'get_torch_model_output_size_at_each_layer type = {type(model)} device = {next(model.parameters()).device}')

    # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
    # We assume that all the model parameters are in a single device
    if not input_tensor:
        assert input_shape != 0
        input_tensor = torch.ones((input_shape), dtype=torch.float32, device=next(model.parameters()).device)

    print('Printing tensor shape after each layer =====================================================')
    print(f'Input Shape = {input_tensor.shape}')
    print('=====================================================')
    for module in model.modules():

        # These two types are redundant. This loop will iterate inside serial object anyway
        if isinstance(module, (nn.Sequential, type(model))):
            continue

        print(module)
        input_tensor = module(input_tensor)
        print(f'Tensor shape = {input_tensor.shape}')
        print('=====================================================')


class ImageDatasetUnsupervisedLearning(Dataset):

    def __init__(self, images_path, image_format='jpg', image_size=None, additional_transforms=None):

        print(images_path + os.sep + '*.' + image_format)
        self.image_files_list = glob.glob(images_path + os.sep + '*.' + image_format)

        pil_to_torch_transforms = get_torch_transforms_pil_image_to_torch_image(new_size=None,
                                                                                minus_1_to_1_scale=False)
        torch_to_pil_transforms = get_torch_transforms_torch_image_to_pil_image(new_size=None,
                                                                                minus_1_to_1_scaled_image=False)

        if image_size:
            pil_to_torch_transforms.append(transforms.Resize(image_size))

        self.train_transforms = transforms.Compose(pil_to_torch_transforms)
        self.display_transforms = transforms.Compose(torch_to_pil_transforms)

        self.additional_transforms = additional_transforms

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        torch_image = self.train_transforms(PIL.Image.open(self.image_files_list[idx]))

        if self.additional_transforms:
            torch_image = self.additional_transforms(torch_image)

        return torch_image

    def show_dataset_image(self, idx):
        image = self.display_transforms(self[idx])
        plt.imshow(image)
        plt.show()

    def show_some_random_images(self, num_images=3):
        for _ in range(num_images):
            idx = random.randint(0, len(self))
            self.show_dataset_image(idx)

    # TODO: Calc std pending
    def get_per_channel_mean_and_std(self, color_images=True):
        channel_sum = [0, 0, 0]
        channel_sqr_sum = [0, 0, 0]

        num_images = len(self)

        if color_images:
            for idx in tqdm(range(num_images)):
                torch_image = self[idx]

                num_pixels = torch_image.shape[1] * torch_image.shape[2]

                r_sum = torch.sum(torch_image[0, :, :]) / num_pixels
                g_sum = torch.sum(torch_image[1, :, :]) / num_pixels
                b_sum = torch.sum(torch_image[2, :, :]) / num_pixels

                channel_sum[0] += r_sum.item()
                channel_sum[1] += g_sum.item()
                channel_sum[2] += b_sum.item()

            channel_sum[0] /= num_images
            channel_sum[1] /= num_images
            channel_sum[2] /= num_images

            print(channel_sum)
