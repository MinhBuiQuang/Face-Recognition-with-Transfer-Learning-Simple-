# Face-Recognition-with-Transfer-Learning-Simple-

## Install project dependencies
```
pip install -r requirements.txt
```
## Project structure
```
-dataset
-models
-FaceData 
---Person1_img
---Person2_img
---Person3_img
.....etc.....
---Unknown
-faceimg
-OpenCV-Haarcascade
---haarcascade_frontalface_default.xml
-create_dateset.py
-train.py
```
## Preparing dataset
For training model, you first need to cut faces from original data to 'faceimg' folder (this path can be modified by passing agrument).
```
python create_dateset.py --image_path ./FaceData
                         --output_face ./faceimg
                         --output_data ./dataset
```
For more detail:
```
python create_dateset.py --help
```
## Training
Once you created X.pickle, y.pickle (faces data and its label) you can start training model as follow:
```
python train.py --data_path ./faceimg
                --output_model ./models                         
```
For more detail:
```
python train.py --help
```
Note: Make sure all the folders exists. I was too lazy to perform some logics in my code, sorry about that!

![alt text](https://github.com/MinhBuiQuang/Face-Recognition-with-Transfer-Learning-Simple-/blob/master/anh1.png?raw=true)
