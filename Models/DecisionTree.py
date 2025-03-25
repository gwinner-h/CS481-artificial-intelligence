'''
This file contains the DecisionTree class, which is used to train a Decision Tree model on a dataset.
'''

from Helpers import Parser, Loader, Evaluator
from sklearn.tree  import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

class DecisionTree:
    def run(self, data=None, dataCSV=None, enablePrinting=False):
        if data is None and dataCSV is None:
            print("Error: No data provided")
            return
        
        loader = Loader()
        if data is None and dataCSV is not None:
            headers, data = loader.loadData(dataCSV)

        # Split the data
        trainX, trainY, textX, testY = loader.splitData(data)
        
        # Search for the best parameter values based on accuracy
        model = DecisionTreeClassifier(random_state=42)
        searchParam = "max_depth"
        searchParam2 = "min_samples_leaf"
        grid = {searchParam: [None] + list(range(1, 11, 1)), searchParam2: range(1, 21, 2)}
        searcher = GridSearchCV(model, grid)
        searcher.fit(trainX, trainY)
        
        if enablePrinting:
            # Print the accuracy for each parameter value checked
            means = searcher.cv_results_['mean_test_score']
            params = searcher.cv_results_['params']
            for mean, param in zip(means, params):
                print(f"{searchParam}: {param[searchParam]}, {searchParam2}: {param[searchParam2]}, Accuracy: {mean}")
            
            # Print the most accurate values
            bestResult = searcher.best_params_[searchParam]
            print(f"The best value of {searchParam} found was {bestResult}")
            bestResult2 = searcher.best_params_[searchParam2]
            print(f"The best value of {searchParam2} found was {bestResult2}\n")
        
        # Test the most accurate model on data it has not seen
        model = searcher.best_estimator_
        predictions = model.predict(textX)
        # Print its test metrics
        if enablePrinting: print(f"Best model results on test data:")

        return Evaluator().getMetrics(testY, predictions, printMetrics=enablePrinting)

if __name__ == "__main__":
    filename = Parser().parseArgs()
    DecisionTree().run(dataCSV=filename, enablePrinting=True)