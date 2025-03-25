'''
This file contains the NearestNeighbors class, which is used to train a model using the Nearest Neighbors algorithm.
'''

from Helpers import Parser, Loader, Evaluator
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

class NearestNeighbors:
    def run(self, data=None, dataCSV=None, enablePrinting=False):
        if data is None and dataCSV is None:
            print("Error: No data provided")
            return
        
        loader = Loader()
        if data is None and dataCSV is not None:
            headers, data = loader.loadData(dataCSV)

        # Split the data
        rows, cols = data.shape
        trainX, trainY, textX, testY = loader.splitData(data)
        
        # Search for the best value of n_neighbors based on accuracy
        model = KNeighborsClassifier()
        searchParam = "n_neighbors"
        grid = {searchParam: range(1, int(np.sqrt(rows) + 3), 2)}
        searcher = GridSearchCV(model, grid)
        searcher.fit(trainX, trainY)
        
        if enablePrinting:
            # Print the accuracy for each value of n_neighbors checked
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
    NearestNeighbors().run(dataCSV=filename, enablePrinting=True)