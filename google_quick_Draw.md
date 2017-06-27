## 1.Detailed description of project:

[Google Quickdraw](https://quickdraw.withgoogle.com/data) released dataset that contains over 50 million drawings on 5/18/2017.
the google quickdraw is online pictionary game application where user is asked to draw a picture of something in 20 seconds. 
While user draws picture, google AI will try to predict what user is drawing.

the dataset that google released contains images and several features related to image.
features include drawing_ID, category(what quickdraw asked to draw), timestamp, whether AI guessed correct or not, user's country and drawing. drawing is represented as a list of list of list.
Drawing is a list of strokes and stroke is a list of X,Y and time (3 lists within a stroke)

With this dataset I ran convolutional neural network and XGboost ensemble method algorithm to determine user's country based on information contained within drawings. unlike for typical CNN image prediction model, the stroke information contains 2 additional dimensions:

|     image       |      stroke        |
|:--------------: | :-----------------:|
|     2D (X,Y )   | 4D(X,Y,time,stroke)|
|     a drawing   | users' process of drawing|


## 2.What question/problem will your project attempt to answer/solve?:
With drawings from multiple categories, I am planning to predict user's country code.

## 3.What technologies / techniques will you use?:
Convolutional neural network, XGboost, random forest

## 4.What are your data sources?:
Google Quick Draw data:
[quickdraw data](https://quickdraw.withgoogle.com/data)
[github page](https://github.com/googlecreativelab/quickdraw-dataset)

