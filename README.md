#Spam example

This is example program demonstrating how textual information can be used for machine learning using scikit-learn
library

Data set is UCI (https://archive.ics.uci.edu/ml/machine-learning-databases/00380/) provided you tube comments
which are classified as 1 = spam , 0 = not spam

spam.py is example program from Book "Python Artificial Intelligence Projects for Beginners" by Joshua Eckroth
it has been slightly modified to run on python 3.x

## How to set up on mac book

1. Install python

```
brew install python3
```

This installs python package manager


2. Clone repository

```
git clone git@github.com:arsalanam/spamexample.git
```

3. run following commands on terminal to create a virtial environment where pandas , numpy and scikit-learn will get
installed

```
cd spamexample
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```
4. run it
```
python spam.py
```

5. review code and comments

documentation links

scikit-learn  working with textual data : http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

scikit-learn Count Vectorizer: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

scikit-learn TDIF Vectorizer :http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html


scikit-learn Ensemble learning and Random forrest:http://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees


