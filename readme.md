## FG2020 Compound Emotion challenge
In this repo, we include the 1st Place code to FG2020 Compound Emotion challenge;

![](https://github.com/Qidian213/Emotion_challenge/blob/master/Net.PNG)

### Introduction
We use [Dlib](https://github.com/davisking/dlib) to do face and landmark detection, and use landmark to do face cropping and alignment, then we use [Pytorch version 1.1](https://github.com/pytorch/pytorch) to with landmark and cropping image to train cnn model to do the face expression recognition task.

#### Pipline

First, you should generate the crop and aligned data on Chanllenge dataset. Change to `crop_align` dir

**Build crop_align**
 ```
 mkdir build
 cd build
 cmake ..
 make
 cd ..
 ```
Dataset path need to be change in `landmark.py` and `crop_align.py`

```
python landmark.py     ### gen train_landmark.json, val_landmark.json, test_landmark.json
python crop_align.py   ### crop face images
python landmark_224.py ### gen landmark_224.json
python prepare_data.py ### gen val_ld.txt, train_ld_shuffle.txt
```
The crop and aligned data of 224x224 will place in `$ROOT/faces_224/`

Then change to `$ROOT` dir, just type
```
python pred_csv.py
python gen_submit.py
```
It will load data and model to generate labels named `predictions.txt` and `predictions.zip` for test data.

**The trained model is just a experiement model, it may not has the best perfomance in this challenge.**
 
Just upload `predictions.zip` to submit window then.

#### Training
Run `sh train_kd.sh` to start training,(10-folder cross-validation).
