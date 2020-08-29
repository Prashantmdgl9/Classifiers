# Imports
import turicreate as tc
import pandas as pd
import matplotlib as plt

# Data Import
sf = tc.SFrame.read_csv('Classifier/yelp-data.csv', header = True)

sf.print_rows(10)
sf.shape
sf.dtype
# Restaurants with rating >=3 are good
sf.groupby('stars', tc.aggregate.COUNT)
# Create a dummy column that will seprate good restaurants from the moderate/bad ones. Rating > 4 is considered good.
sf['is_good'] = sf['stars'] >= 4


# Train and Test split of the data
train, test = sf.random_split(0.8, seed = 1234)
train.shape
test.shape


# We don't know which classifier will suit the data, so let's use the generic one that's provided by graphlab

classifer_model = tc.classifier.create(train, target = 'is_good', features = ['business_review_count', 'business_avg_stars', 'user_review_count' , 'votes_useful', 'user_avg_stars' ])



# Generate predictions ( An interesting observation is that the generic classifier doesn't have the
# other functionalities or associated functions as in the other Classifiers)
predicted = classifer_model.classify(test)

# Evaluate the Logistic Regression model for the predictions it made.
classifier_evaluation = classifer_model.evaluate(test)
classifier_evaluation


# Save the classifer_model
classifer_model.save('Classifier/basic_classifier.model')
