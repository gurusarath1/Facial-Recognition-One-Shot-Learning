COLLECT_FACE_IMAGES_DATA = "COLLECT_FACE_IMAGES_DATA_MODE" # Get user images from webcam for training
PREPROCESS_TRAIN_IMAGES = "PREPROCESS_RAW_TRAIN_DATASET_IMAGES" # Extract only the face portion of the input image dataset
RUN_AUGMENTATION = "AUGMENT_DATASET" # Generate augmented images and store in a folder
TRAIN_LOOP = "TRAIN_MODEL" # Siamese model training
VERIFY_FACE = "FR_MISSION_MODE" # Face verification mode

RUN_MODE = VERIFY_FACE  # MODE OF OPERATION <-------------- Use this constant variable to control code flow
RUN_DEVICE = 'cuda'

TRAINING_IMAGE_SIZE = (80, 80)  # (w,h) # Opposite to x and y
TRAINING_IMAGE_SHAPE = (3, 80, 80)  # torch shape
IMAGE_EXT = '.jpg'

DATASET_DIR = 'G:/Guru_Sarath/Study/1_Project_PhD/git_repos/0_Datasets/Labeled_Faces_in_the_Wild/lfw' #Dir. with the original Labeled_Faces_in_the_Wild dataset
PROCESSED_TRAIN_DATA_DIR = 'processed_train_images' # Dir. where all the cut face images are of Labeled_Faces_in_the_Wild data set is stored
USER_FACE_IMAGES_DIR = 'user_images' # Dir with cut face images of the user

FR_DATASET_DIR = 'final_train_dataset' # Dir. to use for negative images (Labeled_Faces_in_the_Wild faces)
FR_USER_DATASET_DIR = 'final_user_images' # Dir. to use for positive images (Labeled_Faces_in_the_Wild faces)

NUM_EPOCHS = 20
BATCH_SIZE = 500
MODEL_SAVE_PATH = 'saved_models' # Model save path

CONTRASTIVE_LOSS_TRAIN_MARGIN = 10
CONTRASTIVE_LOSS_EVAL_MIN_DIST = 0.5
EVAL_RATIO_SUCCESS_VOTES = 0.7
NUM_TEST_IMAGES_TO_VERIFY = 100

UNKNOWN_USER = 0
USER_VERIFIED = 1