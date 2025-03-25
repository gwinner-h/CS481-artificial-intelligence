'''
This file contains the NeuralNetwork class, which is used to train a Neural Network on a dataset.
'''

from Helpers import Parser, Loader, Evaluator
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

class NeuralNetwork:
    def run(self, data=None, dataCSV=None, enablePrinting=False):
        if data is None and dataCSV is None:
            print("Error: No data provided")
            return
        
        loader = Loader()
        if data is None and dataCSV is not None:
            headers, data = loader.loadData(dataCSV)

        # Split the data
        trainX, trainY, textX, testY = loader.splitData(data)
        
        # Search for the best hidden layer size based on accuracy
        model = MLPClassifier(max_iter=2000, random_state=42)
        searchParam = "hidden_layer_sizes"
        grid = {searchParam: range(1, 11, 1)}
        searcher = GridSearchCV(model, grid)
        searcher.fit(trainX, trainY)
        
        if enablePrinting:
            # Print the accuracy for each hidden layer size checked
            means = searcher.cv_results_['mean_test_score']
            params = searcher.cv_results_['params']
            for mean, param in zip(means, params):
                print(f"{searchParam}: {param[searchParam]}, Accuracy: {mean}")
                
            # Print the most accurate value
            bestResult = searcher.best_params_[searchParam]
            print(f"The best value of {searchParam} found was {bestResult}\n")
            
        # Test the most accurate model on data it has not seen
        model = searcher.best_estimator_
        predictions = model.predict(textX)
        # Print its test metrics
        if enablePrinting: print(f"Best model results on test data:")
        
        return Evaluator().getMetrics(testY, predictions, printMetrics=enablePrinting)


if __name__ == "__main__":
    filename = Parser().parseArgs()
    NeuralNetwork().run(dataCSV=filename, enablePrinting=True)