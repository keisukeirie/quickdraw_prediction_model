# quickdraw_prediction_model
this is my repository for the quick draw prediction model project

## Introduction:

[Google Quickdraw](https://quickdraw.withgoogle.com/data) released dataset that contains over 50 million drawings on 5/18/2017.
the google quickdraw is an application where user is asked to draw a picture of something in 20 seconds and google AI will try to predict what user is drawing within that 20 seconds.

the dataset that google released contains images and features related to that image.
features include object_ID, category(what quickdraw asked to draw), timestamp, whether AI guessed collectly or not and actual drawings.
these drawings are stored as a list of list. It stores strokes as a list and within a stroke, there are lists of X,Y and time data associated with that particular stroke.

 Also note that Unlike image files used on typical neural network image prediction model, drawings from quickdraw contain 1 or 2 more additional dimensions:

  typical image file      |     quickdraw drawing
:------------------------:| :-----------------------:
2D (X,Y) or 3D (X,Y,color)|    4D(X,Y,time,stroke)
      an image            |   how users drew an image


With this dataset I am planning to run neural network to make a prediction model of categories.
for my initial test, I will try to distinguish between drawings of circle and square. If the model is successful, my model will distinguish circle and square based on how users draw pictures.
