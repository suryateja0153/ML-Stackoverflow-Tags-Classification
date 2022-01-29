# Stackoverflow Tags Classification

## Problem Statement
Build a Classification Model in Python that can predict user query tags on Stackoverflow.

## Dataset
Dataset contains the stackoverflow posts with the associated tags in multiple .csv files.<br>

Links to datasets in this .txt file: [Dataset Links](Data/)

Dataset can be found here: [Dataset_1](https://www.kaggle.com/stackoverflow/stacklite) |
[Dataset_2](https://www.kaggle.com/stackoverflow/stacksample)

## Analysis
1. I started with importing the data and applying pre-processing by like combing tables.

2. Used Sklearn librariy for Test-Train split.

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

3. Then, applied feature extraction steps to extract features and vectorize the data.

```
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vect = CountVectorizer(stop_words='english',max_df=.3)
```

4. I used Naive Bayes as it is the simplest algorithm that uses probability of the events for its purpose. It is based on the Bayes Theorem which assumes that there is no interdependence amongst the variables. Calculating these probabilities will help us calculate probabilities of the feature word.

```
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

#fitting the model into train data 
nb.fit(X_train_dtm, y_train)
```

5. Predicting on test and train data and checking the performance metrics.

```
#predicting the model on train and test data
y_pred_class_test = nb.predict(X_test_dtm)
y_pred_class_train = nb.predict(X_train_dtm)

from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class_test))
print(metrics.accuracy_score(y_train, y_pred_class_train))
```

6. Train accuracy is 90% and the test accuracy is 78% which is not bad but can be improved by using advanced classification models.

7. Predict on the test data.

## Conclusion
This was a fun exercise which enabled me to learn more about text pre-processing, feature extraction, applied classification and access model performance based on evaluation metrics. To further improve the model, I would look at SVM's and Deep Learning techniques.