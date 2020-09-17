# Stackoverflow Tags Prediction
Predicting tags on StackOverflow from the 'Title' of the question using classical Machine Learning Models


# Table of Contents:(Testing)
* [Introduction](#Introduction)
* [Data Cleaning](#Data-Cleaning)
* [Exploratory Data Analysis](#Exploratory-Data-Analysis)
* [Text Preprocessing](#Text-Preprocessing)
* [Data Featurization](#Data-Featurization)
* [Model Exploration](#Model-Exploration)
* [Hardware Configuration](#Hardware-Configuration-Used)
* [References](#References)


## Introduction
#### Task:
Predict the tags (a.k.a. keywords), given only the question text and its title. The dataset contains content from disparate stack exchange sites, containing a mix of both technical and non-technical questions.

#### Evaluation Metric:
The evaluation metric for this project is Mean F1-Score. The F1 score, commonly used in information retrieval, measures accuracy using the statistics precision P and recall R. Precision is the ratio of true positives (TP) to all predicted positives (TP + FP). Recall is the ratio of true positives to all actual positives (TP + FN). The F1 score is given by:

> F1=2PR / (P + R)

The F1 metric weights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously. Thus, moderately good performance on both will be favored over extremely good performance on one and poor performance on the other.

The tag predicted must be an exact match, regardless of whether the tags are synonyms.

**Why aren't synonyms counted?**
Giving out a list of candidate synonyms is a potential source of leakage
Synonyms are subjective, and there are "subjectively many" synonyms for a given tag
Equally penalized for predicting a synonym of a correct tag, so the task can be framed as not only predicting a tag, but also modeling the distribution(s) of potential synonyms

**Data Description:**
Train.csv contains 4 columns: Id,Title,Body,Tags

	1. Id - Unique identifier for each question
	2. Title - The question's title
	3. Body - The body of the question
	4. Tags - The tags associated with the question (all lowercase, should not contain tabs '\t' or ampersands '&')
![Data Overview](https://github.com/niteshctrl/stackoverflow-tags/blob/master/images/11.png)

## Data Cleaning
* The data contains 6M datapoints aka rows in total.
* Out of which 1.8M were duplicates. Removing them left 4.2M non-duplicate datapoints which is massive ~30% reduction of data!!
* Seven rows had NULL values in 'Tags' column. Strangely, all of the 'Title' of these seven rows had keywords like 'NULL' or 'Null Objects' with questions like "What is the difference between NULL and '0'?". I proceeded to check if 'NULL' is a tag in the dataset as string but negative. Finally, I decided to remove these rows as 7 out of 4.2M won't make much of a difference.

	
## Exploratory Data Analysis
* Extracted a few rows to view them. It follows that numerals would be of little use in predictions and they will be damaging more than doing good as the population of numerals is very high in Body without significant predictive power.

* The section """<code.>...<./code>""" could be of immense use in predictions as it contains the code of a programming language and it is enough alone to predict the tag(if programing language) but needs to be tested as it could be noisy too owing to very often large codes containing keywords like 'string', 'return', etc which is common to most of the programming languages.

#### Distribution of 'Title Word Length'
![Distn of Title Word Length](https://github.com/niteshctrl/stackoverflow-tags/blob/master/images/1.png)

Most of the titles are in the range of 5-10 words as evident from the distribtion plot. Since, the body is composed of both natural text and programming code, we will skip this analysis on 'Body' as codes could be very lengthly and non-meaningful for analysis

#### Number of Tag Count vs Number of Rows
![Tag Count Distribution](https://github.com/niteshctrl/stackoverflow-tags/blob/master/images/2.png)

Minimum number of tag associated to a row is 1 and maximum is 5. Most of the rows have 3 tags followed by 2 tags and then 4 tags.

#### Unique Tag count(Frequency of Occurence of each Tag)
![Frequency of each tag](https://github.com/niteshctrl/stackoverflow-tags/blob/master/images/3.png)

There are 42048 unique tags in total. Training 42048 models each for a tag will not be an easy task neither scalable too. NOTE: Herefrom, we will be calling c# as Tag 1, java as Tag 2 and so forth in decreasing order of their occurences for ease of plotting.

#### Frequency of Occurence of Tag vs Tag label
![Freq vs Label](https://github.com/niteshctrl/stackoverflow-tags/blob/master/images/4.png)

The above plot shows highly skewed data i.e. most of the datapoints are covered by a few bunch of Tags whose occurence is quite high. We will investigate further if the less occuring tags can be ommited to reduce the pain of training 42K models.

#### Bar Plot of top 20 tags in descending order of occurence
![Bar Plot](https://github.com/niteshctrl/stackoverflow-tags/blob/master/images/6.png)

#### Cummulative Density Function of tags
![CDF](https://github.com/niteshctrl/stackoverflow-tags/blob/master/images/7.png)

It follows that:
![percent conc](https://github.com/niteshctrl/stackoverflow-tags/blob/master/images/8.png)

Just 4699 tags of 42K cover a whopping 90% of the data!!! Training additional 37K models to cover just 10% of the data will be too expensive and we will be ommiting this during modelling.

#### Top 30 Tag strings(Cell) by their frequency of occurence
![top30](https://github.com/niteshctrl/stackoverflow-tags/blob/master/images/9.png)

Observe that most occuring tags cell are of word(tag) length 1 and hence label(tag) dependency is something which can be ignored in this data as the tags are fairly independent.


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


## Hardware Configuration Used:
* Google Colab Notebook with 25GB RAM and ~100GB Storage Space.


## References:
Data Source: https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction

1. https://medium.com/@jorlugaqui/how-to-strip-html-tags-from-a-string-in-python-7cb81a2bbf44
2. https://medium.com/@lukei_3514/dealing-with-contractions-in-nlp-d6174300876b
3. https://pypi.org/project/pycontractions/
4. https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/
5. https://stackoverflow.com/questions/35345761/python-re-split-vs-nltk-word-tokenize-and-sent-tokenize

