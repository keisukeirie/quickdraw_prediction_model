# quickdraw_prediction_model
this is my repository for the quick draw prediction model project  
_last updated: 6/27/2017_  

## Introduction:

[Google Quickdraw](https://quickdraw.withgoogle.com/data) released dataset that contains over 50 million drawings on 5/18/2017.the google quickdraw is online pictionary game application where user is asked to draw a picture of something in 20 seconds.While user draws picture, google AI will try to predict what user is drawing.  
  
  
With this dataset, I wanted to answer following 2 questions:

**_1. Can machine learning models distinguish similar drawings?_**  
**_2. Can machine learning models identify users' country based on their drawings?_**

to answer these questions, I prepared 2 prediction models
1. XGBoost ensemble method model
2. Convolusional Neural Network model

## Instructions

python folder:  
+ contains 3 python files  
  - feature_engineering_func.py:  
    + has python codes to set up json raw file into data that could be used for CNN and ensemble methods
  - CNN_func.py
    + has python codes to run CNN for image recognitions and country prediction
  - ensemble_method_func.py
    + has python codes to run XGboost ensemble method algorithm for image recognitions and country prediction
   
Procedure.ipynb  
+ Jupyter notebook that runs python codes above. note that there is no data stored in this repo.



## Results

**model Accuracy**  

|                 |  image recognition  |   Country prediction   |
|:--------------: | :------------------:|:----------------------:|
|    CNN model    |        88.1%        |           62.7%        |
|  XGBoost model  |        79.1%        |           43.8%        |


## Data used:

the dataset that google released contains images and several features related to image.Features include drawing_ID, category(what quickdraw asked to draw), timestamp, whether AI guessed correct or not, user's country and drawing. drawing is represented as a list of list of list.The drawing feature is a list of strokes and stroke is a list of X,Y and time (3 lists within a stroke)
  
the stroke information contains 2 additional dimensions:

|  typical image  |   Quickdraw data   |
|:--------------: | :-----------------:|
| 3D (X,Y,color ) | 4D(X,Y,time,stroke)|
|     a drawing   | how user drew a drawing|


from this input dataset, I collected image data of **CAT**, **TIGER**, **LION**, **DOG** for image recognition part of my project.  
for country preiction part of my project I selected 4 countries: **US**, **BRASIL**, **RUSSIA** and **SOUTH KOREA**.  
  
I used these 4 countries because these 4 countries had good number of images and they also have different alphabet/language.  
My initial guess was that the way people draw pictures are closely related to how people write.

------------
#### other info:

Image recognition:
+ used 120,000 drawings
  - CAT:30,000  TIGER:30,000  LION:30,000  DOG:30,000
+ drawings were selected randomly

Country prediction:
+ used total of 31,276 drawings from US, Brasil, Russia and South Korea
  - US:8000, BR:8000, RU:8000, KR:7276
+ drawings were filtered by country and selected randomly
+ drawings consists of cat, tiger, lion and dog.


## MODELS
#### Filters applied to both models  
all drawing used in training  
1. were recognized by Google AI correctly
2. contains 15 or less strokes
3. has final time that is 20000ms or less
4. has X and Y ratio where range of Y / range of X =< 1.5
  
**label:**  
image recognition:  
[cat,tiger,lion,dog]  

country prediction:  
[US, BR, RU, KR]
  
-----------------
#### 1. XGBOOST
 Ran codes that creates 399 new features.
 Features include:
 + average number of datapoints per stroke
 + total time spent on drawing
 + time per stroke
 + direction (in radian) of particular stroke
 + stroke number of the stroke with most data points etc. etc. etc.  
 
 
**image recognition model:**  
(max_depth=1, n_estimators=5000, learning_rate=0.25)  
Highest accuracy (6/27/2017): 79.1222222222 percent
  
  
**country prediction model:**  
(max_depth=1, n_estimators=1000, learning_rate=0.2)  
Highest accuracy (6/27/2017): 43.7979539642 percent  
  
-------------------------
#### 2. Convolusion Neural Network Model  
the code I have for CNN applies filtering above and reformat each image into 42 pixel(Y) by 28 pixel(X) format.  
After this process, my CNN data has 1176 columns per image.  

**CNN structure**  
+ 64 convolusion layers with kernel size 5 by 5
+ max pooling layer with pooling size 2 by 2
+ 1 layer of feed forward neural network with 100 neurons
  - relu was used for the activation function 
+ 20% dropout rate was assigned to prevent overfitting
+ final activation function = softmax
+ 4 output neurons  


**Keras parameters and codes:**  
  
```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
from keras.models import load_model

model = Sequential()
model.add(Convolution2D(64, 5, 5, activation='relu', input_shape=(42,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(.20))
model.add(Dense(4, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, nb_epoch=30, verbose=1,validation_split=0.2)
```
  
If you have any suggestion or have better CNN model parameters/code for google quickdraw data, let me know!

--------------

## Findings:  

XG

## OTHER:  
[![**project presentation video DSI capstone project showcase Galvanize Austin 6/22/2017**](http://img.youtube.com/vi/dA4LeDK251A/0.jpg)](https://www.youtube.com/watch?v=dA4LeDK251A)
