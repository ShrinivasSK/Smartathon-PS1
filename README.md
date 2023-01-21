## SDAIA Smartathon
This is Team Young Monks submission for theme 1 of SDAIA Smartathon. 

## YOLO V8 Code

### How to Run
- Add the data files to the `./yolodata` folder (download from [here](https://drive.google.com/drive/folders/1zQOfTEA-5SvU0OUnV_KYqOsLH8wyU6bB?usp=sharing)) and the augmented images in the `./yolodata_aug` folder (download from [here](https://drive.google.com/file/d/19ESN2JFG3LJ3-gf9moutBfhJMQpZ2OCG/view?usp=sharing)).
- Use the train.py to train the model and choose the correct config if you want to train on augmented and non augmented data
- Use the checkpoint and the predict python files to generate the submissions

### File Structure
- [predict.py](predict.py): Use this file to generate predictions from the YOLO Model
- [predict_ensemle.py](predict_ensemble.py): Use this file to generate predictions from the YOLO Model using an ensemble of augmented and normal model
- [train.py](train.py): To Train the YOLO V8 model
- [augmentation.ipynb](augmentation.ipynb): To generate augmented data
- [convert_annotations.ipynb](convert_annotations.ipynb): To convert the train and val csv files to YOLO format to generate annotations file for each image
- [config.yaml](config.yaml) and [config_aug.yaml](config_aug.yaml) Config Files for training YOLO V8

## Faster RCNN Code
Refer to this link [here](https://drive.google.com/drive/folders/1G8Z_hZAyLmFpfYkLJ8KDyiiX1KfBBPby?usp=sharing)