# Stock-price
A practical project for gitchar ML camp

Update on 2019-03-21
------------------------------------------------------------------------
By far using two methods to predice the closing stock price of google.
The first method is using the linear regression. First randomly picking the training and testing data, then treat the open price as the input and the close price as the output. The result seems extremly good. It the criteria of this model was 10% of the error, the precision would be 100%. If the criteria raises to 1%, the precision would be 82.2% percent.

The second method tries to use polynomial feature as the model and then do the regression. At first, still the same, randomly pick up the training and testing data. The input is the date and simplify as 1...n and the output is close stock price. The result was really bad.the error rate would vary from 80% to 90%. No matter how I choose the differenct "degree" value in the model, the result was still bad.
Later, if I got time, I would try to use HMM model to do this project.
