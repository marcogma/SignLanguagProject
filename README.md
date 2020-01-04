# ASL( American Sign Language ) Letters Translator
![](https://d.newsweek.com/en/full/1394686/asl-getty-images.jpg)

### Description
This project aims to implement a real-time video translator of the american sign language.

### Requirements
* Python 3
* OpenCV
* Keras (Latest version)
* Tensorflow
* tqm
* imutils
* Scikit-Learn
* Matplotlib

### How to launch the program
You can run the program in two ways: with the precomputed classifier or by training the classifier before running the program.
#### Launch the program with the precomputed classifier
Before running the project in this mode, make sure that you have the latest version of Kera as the model was trained with the latest version of kera and may have some features not present in older versions. **Note that** the latest version of Keras may not be available on Windows.
**You will have to first download the h5 file at the link <https://drive.google.com/open?id=1lTZdJRL0k8Mk-ptAInDvQibTwgGPKPDV>, and save it to project directory ./savedmodel.**
Then run the following command to launch it with our precomputed model you downloaded and saved it in folder ./savedmodel:
```
python main.py
```
If you want to launch it with an other precomputed model, you need to pass the path to the h5 file. For example:
```
python main.py -m ~/.../othermodel.h5
```
#### Launch the program after training the classifier from scratch
You will have to first download the dataset at the link <https://drive.google.com/open?id=1K9HPAM5tGUCt9O4tH1GGKqpqITpau9u0>, and then you will have to pass the path to saved directory. For example:
```
python main.py -d ~/.../dataset/
```

**Further information on the arguments can be found running the following command:**
```
python main.py -h
```
