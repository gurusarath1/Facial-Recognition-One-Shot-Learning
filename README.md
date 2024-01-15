# Face-Verification
#### Siamese Neural Network / Contrastive Loss

<div align="center">
  <a href="https://www.linkedin.com/in/guru-sarath-t-4ab648131/">
    <img src="https://raw.githubusercontent.com/gurusarath1/gurusarath1/main/includes/images/GitHubLogo_G_anitmation.gif" alt="Guru Sarath T" width="72" height="72">
  </a>
</div>

## Description
I wrote a Python ML code to verify my face. <br>
> DL Architecture - Siamese Neural Network. <be>
> Dataset - Faces in the wild  <br>
> Loss Function - Contrastive Loss <br>
DL model is trained using contrastive loss to make sure my face mapping is far way from other people's face mapping.  <br>

## How to run?
- [x] Install the required dependencies using ```environment.yml``` file
- [x] Run ```main.py```

## How to create your own face ID model?
1. Set the run mode (```RUN_MODE```) in ```face_recognition_settings.py``` file and run ```main.py```.  This will turn on the webcam and collect images of your face. Force stop the code to stop the collection process.
2. Set the run mode (```RUN_MODE```) to ```PREPROCESS_TRAIN_IMAGES```, give the path of the train dataset (```DATASET_DIR```) and run ```main.py``` to cut only the face section of the train images.
3. Optional - Run augmentation on dataset images and your images by setting the (```RUN_MODE```) to ```RUN_AUGMENTATION ``` and running ```main.py```.
4. To train the model set the run mode (```RUN_MODE```) to ```TRAIN_LOOP ``` and run ```main.py```.
5. After training is done, set the run mode (```RUN_MODE```) to ```VERIFY_FACE ``` to test the model.


## Output Screenshots

![output1](/md_support/o1.png) <br><br><br>
![output1](/md_support/o2.png) <br><br><br>
