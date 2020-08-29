In this repository I try to use the basic classifers supplied by Apple's turicreate/graphlab and assess the performance using FPR vs TPR rate

Dataset used = Yelp rating (approx 216K rows)
240 MB size

1. classifier.py is the basic classifier
2. classifier-modified.py is the classifier with a limited feature list
3. LogisticRegression.py is the logistic reg implementation in turicreate lib

basic_classifier.model is the model that can be loaded by the SFrame.

classifier comparison.html is the FPR vs TPR

Bokeh has been used for the visualisation
