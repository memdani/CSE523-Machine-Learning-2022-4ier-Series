# Customer Churn Prediction - Machine Learning 




## Problem Statement
Customer Churn prediction using machine learning. The objective is to test out various classical machine learning algorithms present in order to predict customer churn accurately. It also tries to exhaustively compare algorithms and the effects of data refining on similar algorithms.

**Keywords** - Customer Churn, Classification, Prediction, Logistic Regression, Support Vector Machine, Naive Bayes Classification

## Introduction and Methodology
Customer churn, also referred to as subscriber churn or logo churn, refers to the proportion of subscribers who terminate their subscriptions and is commonly expressed as a percentage. Customer churn prediction and analysis is one of the foremost and widespread applications of classical machine learning. Customer churn is a critical metric that can display customer satisfaction at the macro scale. Additionally, the telecom sector generally sees more significant churn rates than other sectors. This creates a large-scale requirement for better prediction models. 

For the purpose of training the model, the following was implemented in sequential order: 
- Data cleaning: On checking for duplicate and missing values, we found the data accurate and consistent.  
- Exploratory Data Analysis and Data Preprocessing: Conversion of categorical features to numerical features. Trend analysis of each feature with churn rate (y). Data unit conversion where required. 
- Correlation: Correlation matrix to find linear relationships between two variables. 
- Data preprocessing is done and encoding is done.
- Generalized Linear Model: Relations between predictor variables and response variables devised based on the p-values. 
- Feature Scaling: Used to standardise the independent features within a fixed range. 
- Classification Models
For the four models we have used, the approaches are as follows:
1. Binary Logistic Regression
2. Support Vector Machine (SVM)
3. Naive Bayes Classifier
4. Random Forest Classifier
- SMOTE Analysis was done for data balancing.
- Features selection on the basis of correaltion matrix and Principal Component Analysis
- Confusion matrix and accuracy, precision, f1 score and recall were used for model analysis
- Naive Bayes from scratch is observed
- Logistic Regression is analysed by changing parameters and specifications.
- SVM is anaylsed by changing its parameters and choosing the optimal one using GridSearch
- ROC-AUC curve plot are made for analysis. 


## Results
Before and After Data Balancing:

![image](https://user-images.githubusercontent.com/87660206/232231004-7f810551-bccd-41d2-82ef-ee6f382f070d.png)

Feature selection based on correlation and PCA:
![image](https://user-images.githubusercontent.com/87660206/232231173-a0382cd1-6c3f-4858-a8ee-a38f058fd8da.png)

The confusion matrix of models:
![image](https://user-images.githubusercontent.com/87660206/232232625-e845d046-140c-48bd-a4cd-3692e3febae8.png)

![i2](https://user-images.githubusercontent.com/87660206/232232042-9b39b3c2-e457-4cd7-8ff1-364f2ca92a86.png)

Analysis on Logistic Regression on the basis of parameters and specifications:
![image](https://user-images.githubusercontent.com/87660206/232232040-85727731-5927-43c0-9a32-5c696dbbde94.png)

Analysis on SVM on the basis of different parameters and finding the optimal paramters using Grid Search: 

![image](https://user-images.githubusercontent.com/87660206/232232045-53122345-06e7-45e0-9301-3881a64ca6f4.png)


![image](https://user-images.githubusercontent.com/87660206/232232041-3eb3d92b-ac06-4bc3-ba3f-b003a5e2cff2.png)

ROC-AUC Curve:

![image](https://user-images.githubusercontent.com/87660206/232232088-a6112c3b-5129-473f-bcdf-39008f959c4e.png)


![image](https://user-images.githubusercontent.com/87660206/232232092-f5af650b-51dd-45e8-a99c-daccffa06500.png)


![image](https://user-images.githubusercontent.com/87660206/232233324-7e39540a-9bc4-4641-b732-7c39323a4467.png)




## Conclusion
- Smote Analysis was quite effective in our case as we had imbalance in the churn data.
- In the accuracy results of Naive Bayes -> increasing trend
- In the accuracy results of Logistic Regression -> decreasing trend
- After PCA, The final values of accuracy, f1 score, and precision had less impact. 
- For logistic regression, loss function + gradient descent works better.
- For the SVC model, optimal parameters: linear kernel, C=1 & gamma=0.1 are used.
- The AUC value of random forest was maximum (AUC=0.69). 


## References
- [1]P. S, “The A-Z guide to Support Vector Machine,” Analytics Vidhya, Jun. 16, 2021. https://www.analyticsvidhya.com/blog/2021/06/support-vector-machine-better-understanding/
- [2]“Support Vector Machine (SVM) Algorithm - Javatpoint,” www.javatpoint.com. https://www.javatpoint.com/machine-learning-support-vector-machine-algorithm
- [3]“Generalized Linear Models — statsmodels,” Generalized Linear Models — statsmodels. https://www.statsmodels.org/stable/glm.html
- [4]“sklearn.linear_model.LogisticRegression,” scikit-learn. https://scikit-learn/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- [5]“12.1 - Logistic Regression | STAT 462,” 12.1 - Logistic Regression | STAT 462. [Online]. Available: https://online.stat.psu.edu/stat462/node/207/ 
- [6]S. R. Publishing, “Churn Prediction Using Machine Learning and Recommendations Plans for Telecoms,” Churn Prediction Using Machine Learning and Recommendations Plans for Telecoms, Nov. 05, 2019. [Online]. Available: https://www.scirp.org/html/3-1731142_96177.htm 
- [7]“Naive Bayes Classifier in Machine Learning - Javatpoint,” www.javatpoint.com. [Online]. Available: https://www.javatpoint.com/machine-learning-naive-bayes-classifier
