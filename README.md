# Customer-Predictive-Analysis
Predicts money spent by customers using Multilayer Regression and Ensemble training for the Biz Prediction Competition

This is a code written to predict the money spent by customers based on the data set given using Regression methods (normal and ensemble training) available in sklearn library in Python.

*  Please refer to the code for more detailed explanation.

## Pre Conditions:
* Data size was around 35000 samples


## The steps I followed are
### Split the data into 14000, 14000,7000 samples to ensure training of initial layers of regression and then another layer of ensemble training for the first layer. The final 7000 samples were to test the trained model.
* I used SVR, Lasso, Ridge, ElasticNet, BayesianRidge and RandomForestGenerator for the initial 14000 samples. This was the first layer and was trained without normalisation. Normalisation drastically decreased the accuracy of prediction and hence ignored it
* I used Lasso again over this trained models for the next 14000 samples. I used this stacking to help improve increase accuracy and r_2 score while reducing the MAE.
### Prepare and ready dataset and remove all non essential values.
* Changed all categorical values to numerical values, removed and cleaned up data and made modifications to features to improve accuracy.
### Used this model to predict values of the remaining 7000 samples and checked accuracy
* r_2 score was about 0.67. It was predicitng based upon 10 attributes.

## Future possible improvements
* Use neural networks for second layer training. 
* Write custom regressors with better accuracy.

Please feel free to use this code as and how you need it. There is no need to take any permission. For any doubts or bugs regarding the code pls contact me at shanmu9898@icloud.com
