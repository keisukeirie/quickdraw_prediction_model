# Feature engineering notes
**Features**
1. countrycode
2. drawing
3. key_id
4. recognized
5. timestamp
6. word
7. xyt per stroke
8. stroke_number
9. final_time
10. first_X
11. first_Y
12. year
13. month
14. day
15. hour_created
16. total number of datapoints
17. X
18. Y
19. time
20. time per stroke
21. stroke with max time
22. stroke with min time
23. std of time
24. ave datapoints per stroke
25. total time drawing
26. ave time per stroke
27. stroke with max dp
28. stroke with min dp
29. std of dp
30. slope
31. direction

**tested features**
* first_t

##Features
*features from google data*
1. countrycode:
  - label for my project
2. drawing
  - contains x,y,t and stroke info.
  - a list of lists of lists
  - a list of strokes. each stroke contains list of x, y and time.
3. key_id
  - unique id for each image
4. recognized
  - wether drawing was recognized by google AI
5. timestamp
  - timestamp of when drawing was created
6. word
  - category of the pic (ex. cat, dog, rabbit, etc.)

*feature engineered*
7. *xyt per stroke [int]*
  - average number of data points per stroke
  - not finished yet
8. stroke_number [int]
  - total number of stroke
9. final_time [int]
  - time it took google AI to recognize
10. first_X [int]
  - first X value of the first stroke
11. first_Y [int]
  - first Y value of the first stroke
  * first_X and Y identify where user start drawing
12. year [int]
  - year the drawing was created
13. month [int]
  - month the drawing was created
14. day [int]
  - day the drawing was created
**15. hour_created [int]**
  - hour the drawing was created
  - **POSSIBLY A DATA LEAKAGE**
16. total number of datapoints [int]
  - number of data points that exist in a drawing
17. X [list]
  - all position of X
18. Y [list]
  - all position of Y
19. time [list]
  - all data points for time
20. time per stroke [list]
  - time(f) - time(0) for each STROKE
    where:
      * time(f) = last time data point of a stroke
      * time(0) = first time data point of a stroke
21. stroke with max time [int]
  - argmax of time per stroke
22. stroke with min time [int]
  - argmin of time per stroke
23. std of time [float]
  - standard deviation of time per stroke
24. ave datapoints per stroke [float]
  - average number of datapints per stroke
25. total time drawing [int]
  - amount of time user spent drawing
26. ave time per stroke [float]
  - average value of time user took per stroke
27. stroke with max dp [int]
  - stroke number with maximum number of data points
28. stroke with min dp [int]
  - stroke number with minimum number of data points
29. std of dp [float]
  - standard deviation of number of data points
30. slope [float]
  - slope of each stroke
31. direction [float]
  - direction of each stroke (in radian)

##Tested Features
* first_t
  - first t value of the first stroke.
  - mostly zeros
    * for dragons, out of 59884 data points, 2558 data points did not start at time = 0. However 2481 out of 2558 started at time = 1.
    * there were 3 samples that has starting time > 10
  - don't think this tells much.


#notes:
- there are few pictures where total stroke number exceed over 50.
most of them are terrible drawing and not recognized by google AI.
My guess: pen tab or something like pen tab was used to draw these picture.


- max stroke number fixed to 30. that way I can get only 30 columns for any features related to stroke
