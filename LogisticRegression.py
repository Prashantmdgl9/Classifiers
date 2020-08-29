import turicreate as tc

# Load the data
data =  tc.SFrame('yelp-data.csv')

# Restaurants with rating >=3 are good
data['is_good'] = data['stars'] >= 4

# Make a train-test split
train_data, test_data = data.random_split(0.8)

# Create a model.
model = tc.logistic_classifier.create(train_data, target='is_good',
                                    features = ['user_avg_stars',
                                                'business_avg_stars',
                                                'user_review_count',
                                                'business_review_count'])

# Save predictions (probability estimates) to an SArray
predictions = model.classify(test_data)

# Evaluate the model and save the results into a dictionary
results = model.evaluate(test_data)
