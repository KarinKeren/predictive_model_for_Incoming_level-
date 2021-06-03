# predictive_model_for_Incoming_level-
The following repository present predictive Logistic-Regression model for the income level of certain population

# Files:
To open the main code, simply open donor_income.ipynb on any desktop browser, or you can download and run the cells. 

# Overview:
I've asked to build a predictive model for the income level of certain population.
No additional information was provided on the data properties nor features. 

The bilding of the model contained the following steps :
a. EDA
b. Data preparation
c. Modeling
d. Evaluating performance
e. Conclusions

Supervised Algorithms Used:
The main code implemented supervised learning algorithms (Logistic-Regression) in Scikit-learn, evaluated the best algorithm that fit the data, then optimized the classifier using Principal component analysis (Principal component analysis).

# Data

The dataset used in this project is included as income_pred.csv. The modified census dataset consists of approximately 32,000 data points, with each datapoint having 14 features.

# Features

age: Age
workclass: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
education_level: Level of Education (Bachelors, Masters, Doctorate, Assoc-voc,etc)
education-num: Number of educational years completed
marital-status: Marital status
occupation: Work Occupation
relationship: Relationship Status
race: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
sex: Sex
capital-gain: Monetary Capital Gains
capital-loss: Monetary Capital Losses
hours-per-week: Average Hours Per Week Worked
native-country: Native Country
Target Variable

class: Income Class (<=50K, >50K)

# Model results

I've found  maximum accuracy of 0.82 and F1 score of 0.84.

The AUC was found to be 0.87

![image](https://user-images.githubusercontent.com/64970561/120691758-87b46700-c4af-11eb-95f1-a02b84fb9dc7.png)

![image](https://user-images.githubusercontent.com/64970561/120691788-97cc4680-c4af-11eb-976b-e87c260e5989.png)

