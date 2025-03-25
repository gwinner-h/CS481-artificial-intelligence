'''
This script is used to preprocess a dataset.
A new file will be created with the preprocessed data.
'''

from Helpers import Loader, Preprocessor


loader = Loader()
preprocessor = Preprocessor()

headers, data = loader.loadData("./data.csv", printDataDetails=True)
data = preprocessor.dropEmptyRows(data)
# There are no empty columns in the data
# data = preprocessor.dropEmptyCols(data)
headers, data = preprocessor.dropLeastCorrelatedCols(headers, data)
headers, data = preprocessor.dropLowestBetas(headers, data, .001)

loader.saveData("./dataPP.csv", headers, data)