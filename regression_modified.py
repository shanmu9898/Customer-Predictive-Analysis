import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso,Ridge,RandomizedLasso,MultiTaskLasso,ElasticNet,Lars,LassoLars,ARDRegression,BayesianRidge
from sklearn.svm import SVR
import numpy as np
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

#Function to open file and read file name and return dataframe
def openfile(filename):
    f = open(filename)
    df = pd.read_csv(f)
    return df

#Preprocess rows with categorical values
def preprocessrow(featureheader, dataframe):
    featurelist = dataframe[featureheader].tolist()
    labelencoderlist = LabelEncoder()
    labelencoderlist.fit(featurelist)
    labelRating = labelencoderlist.transform(featurelist)
    labelName = featureheader + 'list'
    dataframe[labelName] = pd.Series(labelRating)
    dataframe = dataframe.drop(featureheader, 1)
    return dataframe

#Changes the Birthdays given to age in days (approximate)
def changeBirthdayrow(featureName, df):
    feature = df[featureName].tolist()
    i = 0
    newdate = []
    while(i < len(feature)):
        date = feature[i]
        entries_of_date = str(date).split("/")
        if(len(entries_of_date) <= 1):
            newdate.append(None)
        elif(len(entries_of_date[2]) == 4):
            numberofdays = (12 - (int(entries_of_date[0]) + 1)) * 30
            numberofdays = numberofdays + (30 - int(entries_of_date[1]))
            numberofdays = numberofdays + (2018 - (int(entries_of_date[2]) + 1))*365
            newdate.append(numberofdays)
        else:
            numberofdays = (12 - (int(entries_of_date[1]) + 1)) * 30
            numberofdays = numberofdays + (30 - int(entries_of_date[0]))
            numberofdays = numberofdays + (2018 - (int('19' + entries_of_date[2]) + 1)) * 365
            newdate.append(numberofdays)
        i = i + 1
    df[featureName + ' years'] = newdate

    return df



#Final processing of the Dataframe. Removes Nan and rows which are not required
def preprocessDataSet(name):
    df = openfile(name)
    df = preprocessrow('gender', df)
    df = preprocessrow('member type', df)
    df = preprocessrow('martial status', df)
    df = preprocessrow('store class', df)
    df = preprocessrow('store location', df)
    df = changeBirthdayrow('birthday',df)
    df = df.drop(['observation id', 'visit_date_hour', 'birthday', 'date_join_member', 'store open date', 'member_id', 'store_id', 'store size', 'n_product'] ,1)
    df = df.dropna()
    return df

# Prepare and split into train, secondary train and test sets.
def prepareDataSet(df):
    headers = list(df)

    #Seperating columns
    sperateMoneySpent = headers.index('money_spent')
    datasetMatrix = df.as_matrix()

    #Train with 14000 examples
    datasetTrain = datasetMatrix[:14000]
    datasetTrainWithoutLabels = np.delete(datasetTrain, sperateMoneySpent, 1)
    labels = datasetTrain[:, sperateMoneySpent]

    #Ensemble training data
    datasetTestSecondary = datasetMatrix[14000:24000]
    datasetTestSecondaryWithoutLabels = np.delete(datasetTestSecondary, sperateMoneySpent, 1)
    trueLabelsSecondary = datasetTestSecondary[:, sperateMoneySpent]

    #To measure the accuracy of ensemble trained model
    datasetTestFinal = datasetMatrix[24000:datasetMatrix.shape[0]]
    datasetTestFinalWithoutLabels = np.delete(datasetTestFinal, sperateMoneySpent, 1)
    trueLabelsFinal = datasetTestFinal[:, sperateMoneySpent]



    return headers, datasetMatrix, datasetTrainWithoutLabels, labels, datasetTestSecondary, datasetTestSecondaryWithoutLabels, trueLabelsSecondary, datasetTestFinal, datasetTestFinalWithoutLabels, trueLabelsFinal


# Initial Preprocessing Dataset to open and remove rows and shuffle the dataframe
df = preprocessDataSet('/Users/Bumblebee/PycharmProjects/MachineLearning/train.csv')
df = shuffle(df)

# Trying to normalise Data
# x = df['number squared'].values.reshape(-1,1)
# df['number squared'] = (df['number squared'] - df['number squared'].min())/(df['number squared'].max() - df['number squared'].min())



# Returned values for not normalised and normalised data
headers, datasetMatrix, datasetTrainWithoutLabels, labels, datasetTestSecondary, datasetTestSecondaryWithoutLabels, trueLabelsSecondary, datasetTestFinal, datasetTestFinalWithoutLabels, trueLabelsFinal = prepareDataSet(df)


# Setting up the regressor, pipeline, alphas, and fiting the data
regressor0 = SVR()
regressor1 = Lasso()
regressor2 = Ridge()
regressor3 = ElasticNet()
regressor4 = BayesianRidge()
regressor5 = RandomForestRegressor(n_estimators=20)


# regressor2 = AdaBoostRegressor(,n_estimators=300, random_state=rng)

# alphas = np.arange(1,500)
# steps = [('regressor',regressor)]
# pipeline = Pipeline(steps)
# parameterGrid = dict(regressor__alpha = alphas/10)
# GridSearchResult = sklearn.model_selection.GridSearchCV(pipeline,param_grid=parameterGrid)
# GridSearchResult.fit(datasetTrainWithoutLabels,labels)
regr_0 = regressor0.fit(datasetTrainWithoutLabels,labels)
regr_1 = regressor1.fit(datasetTrainWithoutLabels,labels)
regr_2 = regressor2.fit(datasetTrainWithoutLabels,labels)
regr_3 = regressor3.fit(datasetTrainWithoutLabels,labels)
regr_4 = regressor4.fit(datasetTrainWithoutLabels,labels)
regr_5 = regressor5.fit(datasetTrainWithoutLabels,labels)



# regr_2 = regressor2.fit(datasetTrainWithoutLabels,labels)


# Predicting the Data for normalised version
# predictions = GridSearchResult.predict(datasetTestWithoutLabels)
predictions0 = regr_0.predict(datasetTestSecondaryWithoutLabels)
predictions1 = regr_1.predict(datasetTestSecondaryWithoutLabels)
predictions2 = regr_2.predict(datasetTestSecondaryWithoutLabels)
predictions3 = regr_3.predict(datasetTestSecondaryWithoutLabels)
predictions4 = regr_4.predict(datasetTestSecondaryWithoutLabels)
predictions5 = regr_5.predict(datasetTestSecondaryWithoutLabels)


predictions0_final = regr_0.predict(datasetTestFinalWithoutLabels)
predictions1_final = regr_1.predict(datasetTestFinalWithoutLabels)
predictions2_final = regr_2.predict(datasetTestFinalWithoutLabels)
predictions3_final = regr_3.predict(datasetTestFinalWithoutLabels)
predictions4_final = regr_4.predict(datasetTestFinalWithoutLabels)
predictions5_final = regr_5.predict(datasetTestFinalWithoutLabels)


#
df_enesmble = pd.DataFrame({'predictions0':predictions0, 'predictions1':predictions1,'predictions2': predictions2, 'predictions3': predictions3, 'predictions4': predictions4, 'predictions5': predictions5 })
df_enesmble_matrix = df_enesmble.as_matrix()
regressor_ensemble = Ridge()

regr_ensemble = regressor_ensemble.fit(df_enesmble_matrix, trueLabelsSecondary)



df_enesmble_final = pd.DataFrame({'predictions0_final':predictions0_final, 'predictions1_final':predictions1_final,'predictions2_final': predictions2_final, 'predictions3_final': predictions3_final, 'predictions4_final': predictions4_final, 'predictions5_final': predictions5_final})
df_enesmble_final_matrix = df_enesmble_final.as_matrix()

predictionsfinal = regr_ensemble.predict(df_enesmble_final_matrix)

# predictions2 = regr_2.predict(datasetTestWithoutLabels)


# predictions2 = predictions.reshape(-1,1)
# scl = MinMaxScaler()
# scl.fit_transform(df['money_spent'].values.reshape(-1,1))
# newlabels = scl.inverse_transform(predictions2)


# Print final Data
print(r2_score(trueLabelsFinal,predictionsfinal))
print(mean_absolute_error(trueLabelsFinal,predictionsfinal))

# print(r2_score(trueLabels,predictions2))
# print(mean_absolute_error(trueLabels,predictions2))


# #Find where there is an exceptional error
# for i in range(35857-25001):
#     if(abs(trueLabels[i] - predictions[i]) > 40):
#         print(i)

# Write to a file
df2 = preprocessDataSet('/Users/Bumblebee/PycharmProjects/MachineLearning/test.csv')
df2matrix = df2.as_matrix()
predictionsfortest0 = regr_0.predict(df2matrix)
predictionsfortest1 = regr_1.predict(df2matrix)
predictionsfortest2 = regr_2.predict(df2matrix)
predictionsfortest3 = regr_3.predict(df2matrix)
predictionsfortest4 = regr_4.predict(df2matrix)
predictionsfortest5 = regr_5.predict(df2matrix)

df3 = pd.DataFrame({'predictionsfortest0':predictionsfortest0, 'predictionsfortest1':predictionsfortest1,'predictionsfortest2': predictionsfortest2, 'predictionsfortest3': predictionsfortest3, 'predictionsfortest4': predictionsfortest4, 'predictionsfortest5': predictionsfortest5})
df3matrix = df3.as_matrix()

predictionstestfinal = regr_ensemble.predict(df3matrix)




df_changed = pd.DataFrame(predictionstestfinal)
df_changed.to_csv('/Users/Bumblebee/PycharmProjects/MachineLearning/Output5.csv')



# Plotting Data

# plt.plot(df['number of individual products'],df['money_spent'], '.', color='black')
# plt.plot(range(trueLabels.shape[0]),trueLabels,label="Original Data")
# plt.plot(range(trueLabels.shape[0]),predictions,label="Predicted Data")
# plt.legend(loc='best')
# plt.ylabel('Predicted Money Spent')
# plt.show()
