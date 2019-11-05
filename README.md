 # comp551-assignments

## Logistic Regression & LDA on wine quality prediction and breast-cancer classfication
### Contributor: 
Zijin Nie, Yue Lyu, Simiao Wang
### Abstract: 
In this project we implemented logistic regression and linear discriminant analysis (LDA) models, and  applied these two models to classify wine quality and breast cancer diagnosis. Then we compared the training time and the classification accuracy of these models. The criterion for comparison between the two methods is classification error (percent of incorrectly classified objects; CE). The result was that the logistic regression took significant more time to train and is less accuracy than LDA model. We demonstrated the affect of learning rate on prediction accuracy of logistic regression model. We found that the best learning rate is 0.001. We also observed that with our chosen method of feature selection, it does not improve the accuracy of the two models. Forward feature selection method is used on the wine data set, 4 most relevant features are selected. The square value of theses features were added and improved the model.


## Predicting Subreddit of Reddit Comments Utilizing Voting Emsemble of Several Surpervised Learning Models
### Contributor:
Zijin Nie, Hehuimin Chen, Yue Lyu
### Abstract:
This project developed a supervised-learnin gmodel to classify the comment texts from the popular web forum, Reddit. The final model predicts which subreddit a comment comes from. We compared several natural language feature extraction and data preprocessing methods. Some of the methods were combined andused to select the most relevent features and increase the accurancy of final model. We compared and validated some popular supervised learning classification models and proved that SVM and Multinomial NaïvBayes were more accurate than the ones chosen. During our experiment, we found that voting ensemble (bycombining the output of several models), could be used to construct a comprehensive model, that increased testset accuracy by 1.2% compared to the best single model, Multinomial Naive Bayes.
