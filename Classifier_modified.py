# Imports
import turicreate as tc
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
# Read in Data
sf = tc.SFrame.read_csv('Classifiers/yelp-data.csv', header = True)

sf.print_rows(5)

sf['is_good'] = sf['stars'] >= 4

train, test = sf.random_split(0.8, seed = 1234)
train.shape
test.shape


feature_list = [['business_review_count', 'business_avg_stars', 'user_review_count' , 'votes_useful', 'user_avg_stars' ],
                                                [ 'business_avg_stars', 'user_review_count' , 'user_avg_stars' ],
                                                ['business_review_count', 'user_review_count'],
                                                ['user_avg_stars', 'business_avg_stars'],
                                                ['business_review_count', 'business_avg_stars', 'user_review_count' , 'votes_useful', 'user_avg_stars','votes_cool', 'votes_funny' ]]

models = []
evaluations = []
predictions = []
for list in feature_list:
    model = tc.classifier.create(train, target = 'is_good', features = list)
    predicted = model.classify(test)
    evaluation = model.evaluate(test)
    models.append(model)
    evaluations.append(evaluation)
    predictions.append(predicted)




#Use Bokeh
from bokeh.plotting import figure, output_file, show, output_notebook,curdoc
from bokeh.models import ColumnDataSource, Plot, LinearAxis, Grid, HoverTool, Legend
from bokeh.models.glyphs import MultiLine
from bokeh.palettes import Category10

xs = []
ys = []
for dict in evaluations:
    for key, val in dict.items():
        if(key == 'roc_curve'):
            xs.append(tc.SArray.to_numpy(val['fpr']))
            ys.append(tc.SArray.to_numpy(val['tpr']))

output_file("Classifiers/classifier comparison.html")

p = figure(plot_width=750, plot_height=800, title = "FPR VS TPR for various models")
p.xaxis.axis_label = "FPR"
p.yaxis.axis_label = "TPR"

label_list = []
for x in range(len(feature_list)):
    label_list.append("Model_"+ str(x+1))


data = {'xs': xs,
        'ys': ys,
        'labels': label_list,
        'colors': Category10[len(feature_list)] }

source = ColumnDataSource(data)

p.multi_line(xs='xs', ys='ys', legend='labels', color = 'colors', source=source, line_width = 4)

show(p)
