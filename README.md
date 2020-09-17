# Stackoverflow Tags Prediction
Predicting tags on StackOverflow from the 'Title' of the question using classical Machine Learning Models


# Table of Contents:(Testing)
* [Introduction](#section1)
* [Data Cleaning](#Data-Cleaning)
* [Exploratory Data Analysis](#Exploratory-Data-Analysis)
* [Text Preprocessing](#Text-Preprocessing)
* [Data Featurization](#Data-Featurization)
* [Model Exploration](#Model-Exploration)
* [Areas of Improvement](#Areas-of-Improvement)
* [Hardware Configuration](#Hardware-Configuration-Used)
* [References](#References)

## Data Cleaning
* The data contains 6M datapoints aka rows in total.
* Out of which 1.8M were duplicates. Removing them left 4.2M non-duplicate datapoints which is massive ~30% reduction of data!!
* Seven rows had NULL values in 'Tags' column. Strangely, all of the 'Title' of these seven rows had keywords like 'NULL' or 'Null Objects' with questions like "What is the difference between NULL and '0'?". I proceeded to check if 'NULL' is a tag in the dataset as string but negative. Finally, I decided to remove these rows as 7 out of 4.2M won't make much of a difference.

	
## Exploratory Data Analysis
* Extracted a few rows to view them. It follows that numerals would be of little use in predictions and they will be damaging more than doing good as the population of numerals is very high in Body without significant predictive power.
* The section """<code.>...<./code>""" could be of immense use in predictions as it contains the code of a programming language and it is enough alone to predict the tag(if programing language) but needs to be tested as it could be noisy too owing to very often large codes containing keywords like 'string', 'return', etc which is common to most of the programming languages.
#### * Distribution of 'Title Word Length'
![Distn of Title Word Length](https://github.com/niteshctrl/stackoverflow-tags/blob/master/images/1.png)


## Text Preprocessing
* General tasks performed:
  1. HTML tags Removal
  2. Punctation Removal
  3. Stopwords Removal
  4. Stemming(Used SnowBall Stemmer: This also converts the tokens to lowercase).
  5. Expanding Contractions
  6. Removed Numerals from 'Body'
  
* Specific to this project:
  1. Repeated Uppercase tokens to increase their weightage.
  * Why?
   We generally tend to write the keyowords in uppercase on stackoverflow for better visibility.
  Ex - HTML vs html, CSS vs css, C vs c, PHP vs php, and the list goes on. These are the keywords that are to be predicted and hence more weightage.
								
								
## Data Featurization
### Steps Used:
* Split data before vectorization to avoid data leakage.
* Used Cross Validation Set as 20% of the entire data as the dataset was faily large and 20% could accomodate good number of datapoints avoiding any bias.
* Used **Binary Count Vectorizer** for defining the presence of a tag in a row of data. This will be the output label in sparse matrix format 
  * [0 1 0 0 0 0 1 . . . . . .  0 0 0] where '1' represents the presence of corresponding Tag.
* Used **Tf-IDF Vectorizer** with 100K max features to vectorize the input features aka text.
* Used n_gram range = (1,1) as predicting Tags would be enough by a single word.
  * For example: In title : "Why C++ is faster than python?", we don't need 'Why C++' or 'C++ is' to predict the tag 'C++'.

### Unused features:
* Defined 'title_weight' as function parameter because titles are more useful in predicting tags considering the word length of 'Title' vs 'Body'.
* Vectorizing both Title and Body and merging them later might be more beneficial as this will increase the dimensionality and sparsity of data as opposed to merging first and then vectorizing. But then, merging them first will enable us to put more weight on the Title part than the Body which could be positive point for the models accuracy.


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


## Areas of Improvement:


## Hardware Configuration Used:
* Google Colab Notebook with 25GB RAM and ~100GB Storage Space.

## References:
1. https://medium.com/@jorlugaqui/how-to-strip-html-tags-from-a-string-in-python-7cb81a2bbf44
2. https://medium.com/@lukei_3514/dealing-with-contractions-in-nlp-d6174300876b
3. https://pypi.org/project/pycontractions/
4. https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/
5. https://stackoverflow.com/questions/35345761/python-re-split-vs-nltk-word-tokenize-and-sent-tokenize

