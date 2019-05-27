# Nail Classifier

CNN classifier for quality control of nails. To classify good and bent nails.
Transfer learning technique is adopted. Since the dataset is small and is different from the public domain datasets feature extraction of vgg16 and vgg19 is adopted. Two diffrent methods are tested.
* extracting features from the convolutional base layer of VGG network and training a custom classifier on top of it.(refer nail-classifier.py)
* extracting the features of the first 3 layers (earlier layers) of VGG network and training a custom classifier on top of it.(refer vgg16-early-extract.py)

## Training

### Split the dataset

To split the dataset to train, validate and test

```
# cd to working directoy
pip install split-folders
# split the train, validate and test in ratio 0.8:0.1:0.1
split_folders <folder/path/with/images> --output <output/folder/path> --ratio .8 .1 .1
```
### Train


```
cd <working-directory>
git clone https://github.com/amruz/nail-classification.git
cd nail-classification
# Change the input arguments if needed in config.ini file eg: dataset path 
python3 nail-classifier.py

``` 

### Predictions as web service 

 Docker containerized REST API

```
# Docker build and run 
sudo docker build -t nail-classifier .
sudo docker run -d -p 3000:3000 nail-classifier
# Query
curl http://<DOCKER-IP>:3000/predict?image_url=http://domain.com/image.jpeg
# File
curl -X POST -F image=@image.jpeg 'http://<DOCKER-IP>:3000/predict'
```
