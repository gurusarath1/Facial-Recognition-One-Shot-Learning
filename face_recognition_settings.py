COLLECT_FACE_IMAGES_DATA = "COLLECT_FACE_IMAGES_DATA_MODE"
VERIFY_FACE = "FR_MISSION_MODE"
PREPROCESS_TRAIN_IMAGES = "PREPROCESS_RAW_TRAIN_DATASET_IMAGES"
RUN_AUGMENTATION = "AUGMENT_DATASET"
TRAIN_LOOP = "TRAIN_MODEL"

RUN_MODE = VERIFY_FACE  # MODE OF OPERATION
RUN_DEVICE = 'cuda'

TRAINING_IMAGE_SIZE = (80, 80)  # (w,h) # Opposite to x and y
TRAINING_IMAGE_SHAPE = (3, 80, 80) # torch shape
IMAGE_EXT = '.jpg'

DATASET_DIR = 'G:/Guru_Sarath/Study/1_Project_PhD/git_repos/0_Datasets/Labeled_Faces_in_the_Wild/lfw'
PROCESSED_TRAIN_DATA_DIR = 'processed_train_images'
USER_FACE_IMAGES_DIR = 'user_images'

FR_DATASET_DIR = 'final_train_dataset'
FR_USER_DATASET_DIR = 'final_user_images'


NUM_EPOCHS = 1000
BATCH_SIZE = 100
MODEL_SAVE_PATH = 'saved_models'