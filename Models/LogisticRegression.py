'''
This file contains the LogisticRegressor class, which is used to train a Logistic Regression model on a dataset.
'''

from sklearn.model_selection import GridSearchCV
from Helpers import Parser, Loader, Evaluator
from sklearn.linear_model import LogisticRegression

class LogisticRegressor:
    def run(self, data=None, dataCSV=None, enablePrinting=False):
        if data is None and dataCSV is None:
            print("Error: No data provided")
            return
        
        loader = Loader()
        if data is None and dataCSV is not None:
            headers, data = loader.loadData(dataCSV)

        # Split the data
        trainX, trainY, textX, testY = loader.splitData(data)
        
        # Train a logistic regressor to classify data
        model = LogisticRegression(random_state=42, max_iter=2000)
        searchParam = "solver"
        grid = {searchParam: ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
        searcher = GridSearchCV(model, grid)
        searcher.fit(trainX, trainY)
        
        if enablePrinting:
            # Print the accuracy for each solver checked
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
    LogisticRegressor().run(dataCSV=filename, enablePrinting=True)