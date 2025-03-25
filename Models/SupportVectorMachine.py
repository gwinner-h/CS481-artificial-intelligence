'''
This file contains the SupportVectorMachine class, which is used to train a SupportVectorMachine on a dataset.
'''

from Helpers import Parser,Loader, Evaluator
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class SupportVectorMachine:
    def run(self, data=None, dataCSV=None, enablePrinting=False):
        if data is None and dataCSV is None:
            print("Error: No data provided")
            return
        
        loader = Loader()
        if data is None and dataCSV is not None:
            headers, data = loader.loadData(dataCSV)

        # Split the data
        trainX, trainY, textX, testY = loader.splitData(data)
        
        # Search for the best kernel
        model = SVC(random_state=42)
        searchParam = "kernel"
        grid = {searchParam: ["linear", "poly", "rbf", "sigmoid"]}
        searcher = GridSearchCV(model, grid)
        searcher.fit(trainX, trainY)
        
        if enablePrinting:
            # Print the accuracy for each kernel checked
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
    SupportVectorMachine().run(dataCSV=filename, enablePrinting=True)