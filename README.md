# quickdraw_prediction_model
this is my repository for the quick draw prediction model project

## Introduction:

[Google Quickdraw](https://quickdraw.withgoogle.com/data) released dataset that contains over 50 million drawings on 5/18/2017.the google quickdraw is online pictionary game application where user is asked to draw a picture of something in 20 seconds.While user draws picture, google AI will try to predict what user is drawing.  
  
  
With this dataset, I wanted to answer following 2 questions:

**_1. Can machine learning models distinguish similar drawings?_**  
**_2. Can machine learning models identify users' country based on their drawings?_**

to answer these questions, I prepared 2 prediction models
1. XGBoost ensemble method model
2. Convolusional Neural Network model

## Results

**model Accuracy**  

|                 |  image recognition  |   Country prediction   |
|:--------------: | :------------------:|:----------------------:|
|    CNN model    |        88.1%        |           62.7%        |
|  XGBoost model  |        79.5%        |           43.8%        |


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
Highest accuracy (6/27/2017): percent
  
  

**country prediction model:**  
(max_depth=1, n_estimators=1000, learning_rate=0.2)  
Highest accuracy (6/27/2017): 43.7979539642 percent  
  
  
