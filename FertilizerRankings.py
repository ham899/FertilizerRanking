# File: FertilizerRankings.py
# Author: Hunter Moricz
# Last Modified: June 30, 2025
# Description: Trains a CatBoost model on the full competition dataset using tuned 
# hyperparameters, and generates ranked fertilizer predictions for the test set. 
# Saves a CSV file containing top-3 fertilizer recommendations for each test sample 
# for competition submission.
# --------------------------------------------------------------------------------------------

################################### Libraries and functions ##################################

import json
import time
import math

# Data Science Libraries
import numpy as np
import pandas as pd

# Data Preparation
from sklearn.preprocessing import LabelEncoder

# Modelling
from catboost import CatBoostClassifier

# Custom Functions -> Function definitions can be found in utils.py file
from utils import list_to_string, generate_model_rankings


######################################## Set Up ##############################################

# Set the seeds
sampler_seed = 346346
split_seed = 4326
model_seed = 36209436

# Read in the training data
train = pd.read_csv('Data/train.csv', index_col='id')
train = train.rename(columns={'Temparature':'Temperature'})

# Store the numeric and categorical column names
numeric_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_cols = ['Soil Type', 'Crop Type']

# Save the feature columns
features = numeric_cols + categorical_cols

# Extract the target column name
target = 'Fertilizer Name'

# Get the categorical column indices
cat_cols_idxs = [int(inx) for inx in np.where(np.isin(features, categorical_cols))[0]]

# Fit LabelEncoder to the target
le = LabelEncoder()
le.fit(train[target])

# Save the mapping of the label encoding in list form
class_mapping = list(le.classes_)

# Read the tuned hyperparameters from the file they are stored in
with open('Study_Results/Phase2/best_params.json', 'r') as file:
    hyperparameters = json.load(file)

# Remove od_wait and od_type parameters as we are not using early stopping
hyperparameters.pop('od_wait')
hyperparameters.pop('od_type')

# Print the hyperparameter values for verification
print('\nHyperparameters:')
for parameter, setting in hyperparameters.items():
    print(f'\t- {parameter}: {setting}')
print('')

# Get the features and target data
X = train[features]
y = train[target]

# Encode the target with LabelEncoder
y_encoded = le.transform(y)


##################################### Modelling ##############################################

# Get the estimated number of iterations needed
with open('average_iterations.txt', 'r') as file:
    iterations = int(file.read())

# Since the final model has more data than what was trained to obtain the `iterations` value,
# we should scale iterations by a factor to slightly increase the number of iterations needed.
# A conventional scaling factor for boosting models is to take sqrt(1/train_split), which
# for us is sqrt(1/0.75).
iterations = math.sqrt(1/0.75) * iterations
# Ensure iterations are rounded to an integer
iterations = int(round(iterations))

# Initialize the CatBoostClassifier model with the best parameters
model = CatBoostClassifier(
    **hyperparameters,
    iterations = iterations,
    allow_writing_files=False,
    random_seed = model_seed
)

# Fit the model to the training data, track the runtime, and print the results
print('-------------------------------------------')
print('Initiate CatBoost Model Fitting')
print(f'Number of iterations: {iterations}')
print('-------------------------------------------')
start = time.time()
model.fit(
    X, 
    y_encoded, 
    cat_features=cat_cols_idxs,
    verbose=True
)
end = time.time()

# Print the runtime
print(f'\nFinal Model Fitting Runtime: {(end-start)/60:.2f} minutes.\n')


######################################### Rankings ############################################

# Read in the test data
test = pd.read_csv('Data/test.csv', index_col='id')
test = test.rename(columns={'Temparature':'Temperature'})

# Generate the model prediction rankings on the test data
rankings = generate_model_rankings(model, test[features], class_mapping)

# Add the rankings to the DataFrame
test['Fertilizer Name'] = rankings

# Convert the rankings from array format to a string
test['Fertilizer Name'] = list_to_string(test['Fertilizer Name'])


######################################### Submission ##########################################

# Get the DataFrame of only the rankings
submission = test[['Fertilizer Name']]

# Save the rankings to a CSV file for submission
submission.to_csv('Output/Fertilizer Rankings.csv')

# Print the final ranking results generated for view
print(submission.head(20))
