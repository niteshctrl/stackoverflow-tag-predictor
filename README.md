# Stackoverflow Tags Prediction
Predicting tags on StackOverflow from the 'Title' of the question using classical Machine Learning Models


# Table of Contents:(Testing)
* [Introduction](#section1)
* [Exploratory Data Analysis](#Exploratory-Data-Analysis)
* [Data Featurization](#Data-Featurization)
* [Model Exploration](#Model-Exploration)
* [Hardware Configuration](#Hardware-Configuration-Used)
* [References](#References)

## Exploratory Data Analysis

## Data Featurization
### Steps:
* Split data before vectorization to avoid data leakage.
* Used Binary Count Vectorizer for defining the presence of a tag in a row of data. This will be the output label in the format
                                          * [0 1 0 0 0 0 1 . . . . . .  0 0 0]
* Used Tf-IDF Vectorizer with 100K max features


## Model Exploration
### Results
* Trained Logistic Regression Model with GridSearch to tune Hyperparameters on 500K datapoints and 700 Tags.
* It gave a **Micro-F1 Score of 0.474** with alpha = 0.01 as Regularization Parameter.

### RoadMap Used:
**Logistic Regression**
* Initial attempt to train 75 Models on 4.2M datapoints for 75 Output labels was successful resulting with **Micro-F1 Score of 0.59**.
* Tried the same including the 'Body' alongwith the 'Title' but Micro-F1 of 0.04 directed me to ditch and stick to 'Title' only. Probably Body had a lot of noise even after my preprocessing and featurization. Owing to time constraint, I decided to ditch 'Body' and stick to 'Title' only.
* Failed Attempts:
  * Next attempt was to train 5k Models for 5000 Tags but had 'Memory Error'.
  * Reduced the TFIDF features to 100K from 240K but still the same error.
  * Next tried with 1M datapoints instead of 4.2M but 'Memory Error' again.
  * Tried 1500 Models for 1500 Tags and it skipped 'Mem Error' this time resulting in Micro-F1 of 0.3 without Hyperparameter tuning but took ~1Hour. 
  * Gave a final try to train 1500 Models with 4.2M datapoints and it failed again due to 'Memory Error'.
* Finally I settled with training 700 Models for top 700 tags with 500K datapoints.

**Support Vector Classifier**
* Train SVC on 50K datapoints to get **Micro-F1 = 0.438** for 75 Tags without Hyperparameter Tuning.

Although the Micro-F1 score of 0.438 looks decent without hyperparameter tuning, but training only 75 Tags alone took more than 3 Hours which is very costly. Hence terminated Model exploration as non-linear models, though with better results,  will be extermely costly and hence restricting to only to linear models(Logistic Regression).


## Hardware Configuration Used:

## References:
