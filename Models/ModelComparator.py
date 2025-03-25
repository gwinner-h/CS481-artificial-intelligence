'''
This file is used to compare the performance of different models
on a dataset with different preprocessing techniques applied to it.
'''

import numpy as np
from Helpers import Preprocessor, Loader
from DecisionTree import DecisionTree
from LogisticRegression import LogisticRegressor
from NearestNeighbors import NearestNeighbors
from NeuralNetwork import NeuralNetwork
from SupportVectorMachine import SupportVectorMachine

prep = Preprocessor()

# Original data
headers, data = Loader().loadData("./data.csv")

# Data with no empty rows
dataNER = prep.dropEmptyRows(data)
# Data without the least correlated columns
headersNLCC, dataNLCC = prep.dropLeastCorrelatedCols(headers, data)
# Data without the highest beta columns
headersNHB, dataNHB = prep.dropLowestBetas(headers, data, .0005)

# Data with no empty rows and without the least correlated columns
headersNER_NLCC, dataNER_NLCC = prep.dropLeastCorrelatedCols(headers, dataNER)
# Data with no empty rows and without the highest beta columns
headersNER_NHB, dataNER_NHB = prep.dropLowestBetas(headers, dataNER, .0005)
# Data without the least correlated columns and without the highest beta columns
headersNLCC_NHB, dataNLCC_NHB = prep.dropLowestBetas(headersNLCC, dataNLCC, .001)

# Fully preprocessed data
headersPP, dataPP = prep.dropLowestBetas(headersNER_NLCC, dataNER_NLCC, .001)

models = [DecisionTree(), LogisticRegressor(), NearestNeighbors(), NeuralNetwork(), SupportVectorMachine()]
datasets = {
    'unmodified data': data,

    'data with no empty rows': dataNER,
    'data w/o least correlated columns': dataNLCC,
    'data w/o lowest beta columns': dataNHB,

    'data with no empty rows and w/o least correlated columns': dataNER_NLCC,
    'data with no empty rows and w/o lowest beta columns': dataNER_NHB,
    'data w/o least correlated columns or lowest beta columns': dataNLCC_NHB,

    'fully preprocessed data': dataPP
}

metric = "Accuracy"

modelDatasetScores = np.zeros((8, 5))
# Run each model on each dataset
bestCombo = ("", "", 0) # Initialize with a low value
worstCombo = ("", "", float('inf'))  # Initialize with a high value
for i, model in enumerate(models):
    modelName = model.__class__.__name__
    for j, (dataname, dataset) in enumerate(datasets.items()):

        print(f"Running {modelName} model on {dataname}")
        metrics = model.run(data=dataset, enablePrinting=False)
        print(f"Model Metrics: Accuracy: {metrics['Accuracy']}, Recall: {metrics['Recall']}, Precision: {metrics['Precision']}, Specificity: {metrics['Specificity']}, F-Stat: {metrics['FStat']}\n")
    
        modelDatasetScores[j][i] = metrics[metric]
        
        if metrics[metric] < worstCombo[2]:
            worstCombo = (modelName, dataname, metrics[metric])

        if metrics[metric] > bestCombo[2]:
            bestCombo = (modelName, dataname, metrics[metric])
    print()


print('\n\n\n')
bestModel = ("", 0) # Initialize with a low value
worstModel = ("", float('inf'))  # Initialize with a high value
for i, model in enumerate(models):
    # Get the name of the model and its average accuracy
    modelName = model.__class__.__name__
    modelAverageScore = np.mean(modelDatasetScores[:, i])

    # Print the average accuracy for each model
    print(f"Average accuracy for {modelName}: {modelAverageScore}")
    
    # If this model is the worst so far, update the worst model
    if modelAverageScore < worstModel[1]:
        worstModel = (modelName, modelAverageScore)

    # If this model is the best so far, update the best model
    if modelAverageScore > bestModel[1]:
        bestModel = (modelName, modelAverageScore)

# Print the most and worst model
print(f"The worst model is {worstModel[0]} with an average {metric} of {worstModel[1]}")
print(f"The best model is {bestModel[0]} with an average {metric} of {bestModel[1]}")
print()

bestDataset = ("", 0) # Initialize with a low value
worstDataset = ("", float('inf'))  # Initialize with a high value
for j, (dataname, dataset) in enumerate(datasets.items()):
    # Get the name of the dataset and its average accuracy
    datasetAverageScore = np.mean(modelDatasetScores[j, :])

    # Print the average accuracy for each dataset
    print(f"Average {metric} for {dataname}: {datasetAverageScore}")
        
    # If this dataset is the worst so far, update the worst dataset
    if datasetAverageScore < worstDataset[1]:
        worstDataset = (dataname, datasetAverageScore)

    # If this dataset is the best so far, update the best dataset
    if datasetAverageScore > bestDataset[1]:
        bestDataset = (dataname, datasetAverageScore)

# Print the most and worst datasets
print(f"The worst dataset is {worstDataset[0]} with an average {metric} of {worstDataset[1]}")
print(f"The best dataset is {bestDataset[0]} with an average {metric} of {bestDataset[1]}")
print()
print(f"The worst combination of model and dataset is {worstCombo[0]} on {worstCombo[1]} with {metric}: {worstCombo[2]}")
print(f"The best combination of model and dataset is {bestCombo[0]} on {bestCombo[1]} with {metric}: {bestCombo[2]}")

Loader().saveData("./ModelComparison.csv", ["Decision Tree", "Logistic Regression", "Nearest Neighbors", "Neural Network", "Support Vector Machine"], modelDatasetScores)