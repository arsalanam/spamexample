This is example program demonstrating processing textual information for machine learning using scikit-learn
library

Data set is UCI (https://archive.ics.uci.edu/ml/machine-learning-databases/00380/) provided you tube comments
 which are classified as 1 = spam , 0 = not spam

spam.py is example program from Book "Python Artificial Intelligence Projects for Beginners" by Joshua Eckroth
it has been slightly modified to run on python 3.x

How to set up on mac book

1. Install python

brew install python3

This installs python package manager


Clone repository

2. git clone spamexample


3. run following commands on terminal to create a virtial environment where pandas , numpy and scikit-learn will get
installed

cd spamexample
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

4. run it ...python spam.py

5. review code and comments




