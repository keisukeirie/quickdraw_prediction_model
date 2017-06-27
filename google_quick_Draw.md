## 1.Detailed description of project:

[Google Quickdraw](https://quickdraw.withgoogle.com/data) released dataset that contains over 50 million drawings on 5/18/2017.
the google quickdraw is an application where user is asked to draw a picture of something in 20 seconds and google AI will try to predict what user is drawing within that 20 seconds.

the dataset that google released contains images and features of that images.
features include ID, category(what it asked to draw), timestamp, whether AI guessed it collect or not and stroke. the stroke feature contains time, x and y axis of each stroke of a drawing.

With this dataset I am planning to run a neural network to determine user's country code based on stroke information. unlike input images for typical neural network image prediction model, the stroke information contains 2 more additional dimensions:

    image      |     stroke
:--------------: | :----------------:
    2D (X,Y)   | 4D(X,Y,time,stroke)
   a drawing   | users' process of drawing

moreover, unlike typical image prediction model, I am training program how users are drawing their picture; not picture's characteristic (e.g. number of strokes, direction of drawing, how user draw a shape etc.).

## 2.What question/problem will your project attempt to answer/solve?:
With this stroke data from multiple categories, I am planning to predict user's country code (I will start off with 4 countries. if it is too easy to predict, will increase number of nations for my label).

## 3.What data will you need to answer the question / solve the problem?:
1. select multiple categories from entire dataset (there are over 300 categories and each categories contains over 100,000 images)
2. I will then gather data from 4 different countries (1 from Asia, 1 from Europe, 1 from South America and US) for each categories
3. based on strokes, x,y coordinates and time information of images, I will try to predict country code (1 out of 4). A simple classifier. simple. I hope.

## 4.What technologies / techniques will you use?:
probably (and unfortunately) neural network.

## 5.How will you present your work?:
- I am thinking of presenting my work in a form of multiple choice questions.
- I will display drawings on my presentation slides and I will let my audience guess the country of artist (4 choices)
- after audience guess their answer, I will display probability that came out of my model
- and real answer.
- I believe this might be a good way to show how effective my model is to predict country.

## 6.Web app - where will you host it, what kind of information will you present, and what technologies will you use to build it?:
If I can smoothly create my model and have enough time to create a web app, I would like to have a website that displays my results in a form of multiple choice question like the step.5 above.
1. website randomly displays pictures from same country code.
2. visitors can guess the country (multiple choice question with 4 choices)
3. model will calculates probability for each choice
4. displays real answer

technology: probably flask.

## 7.Visualization - what final visuals are you aiming to produce?:
if I have enough time to show my work, probably have several plots to show accuracy of my model.
Additional idea: show how much model improves by including additional category in my model training data.

## 8.Presentation - slides, interpretive dance?:
powerpoint slides. If you know any good powerpoint like app that you recommend using, please let me know.

## 9.What are your data sources?:
Google Quick Draw data:
[quickdraw data](https://quickdraw.withgoogle.com/data)
[github page](https://github.com/googlecreativelab/quickdraw-dataset)

## What is the next thing you need to work on?:

1. get data (specifically look at country codes... how many countries? how are they distributed?) and restudy keras and neural network.
2. formulate project plans/procedures
3. create a model
4. test model
5. repeat 3. and 4.
6. work on presentation/ visualization aspect of models

## Acquiring the data (if you haven't already):
I can download data from github page.

## Understanding the data?:
I looked at 4 categories. it looks like most data came from the US, Canada, GB (English speaking countries). I realized that the quickdraw website was not translated into different language.

most likely, I need to under-sample data coming from the US.

## Building a minimum viable product?:

## Gauging how much signal might be in the data?:
