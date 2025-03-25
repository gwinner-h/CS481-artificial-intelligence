'''
This script is used to find the best value of alpha for a Lasso model.
'''

from Helpers import Parser, Loader
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


datafile = Parser().parseArgs()
headers, data = Loader().loadData(datafile, printDataDetails=False)
# Split the data
trainX, trainY, textX, testY = Loader().splitData(data)

model = Lasso(random_state=42)
searchParam = "alpha"
grid = {searchParam: [.0001, .0005, .001, .005, .01, .05, .1, 1, 5, 10, 50, 100, 500, 1000]}
searcher = GridSearchCV(model, grid)
searcher.fit(trainX, trainY)

# Print the best alpha value found
bestResult = searcher.best_params_[searchParam]
print(f"The best value of {searchParam} found was {bestResult}\n")