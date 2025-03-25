'''
This file contains several helper classes and functions used by the model training scripts.
'''

import os, argparse, numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.src.layers import Conv2D


class Parser: 
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="The path to the file with your data",
                        type=str, nargs='?', default="./data.csv")
    
    def parseArgs(self):
        '''
        Return:
            A string containing the path to the data file
        '''
        args = self.parser.parse_args()
        file = args.datafile

        if os.path.exists(file):
            return file
        else:
            print(f"Error: {file} does not exist.")
            exit(1)
    


class Loader:
    # Load data from a csv
    def loadData(self, fname, printDataDetails=False):   
        '''
        Parameters:
            fname: a string containing the path to your data csv file
            printData: whether or not to print information about the data loaded
        Return:
            An array of headers (feature names)\n
            A 2d array containing your data without headers
        ''' 
        print("Loading Data..........")
        # Load the data
        arr = np.loadtxt(fname, delimiter=",", dtype=str)
        headers = arr[0, :]
        data = np.delete(arr, 0, 0)

        data = data.astype(int)
        
        # Print information about the data loaded
        if printDataDetails:
            print()
            print(f"Feature Names: \n{headers}\n")
            print(f"Data Loaded: \n{data}\n")
            print(f"Target Data: \n{data[:, -1]}\n")
            print(f"Correlation Coefficients:\n{np.corrcoef(np.transpose(data))[-1]}\n")
            print(f"Target Name: {headers[-1]}\n")
        # Always say the shape of the data
        print("Data Loaded!")
        print(f"Data Shape: {arr.shape}")
        print()
        
        return headers, data
    
    def saveData(self, fname, headers, data):
        '''
        Parameters:
            fname: a string containing the path to your data csv file
            data: a 2d array containing your data
        Function:
            Save data to a csv file
        '''
        data = np.vstack((headers, data))

        print("\nSaving Data..........")
        np.savetxt(fname, data, delimiter=",", fmt="%s")
        print("Data Saved!")
    
    # Split data into training and test predictors and outcomes
    def splitData(self, data, test_size=.2):
        '''
        Parameters:
            data: A 2d array containing your data
            test_size: the percentage of data that should be reserved for testing
        Return:
            The predictors to be used in model training\n
            The outcomes to be used in model training\n
            The predictors to be used in model testing\n
            The outcomes to be used in model testing
        ''' 
        # Split the data into training and test sets
        train, test = train_test_split(data, random_state=42, test_size=test_size)

        # Separate the predictors from the outcome
        trainX = train[:, :-1]
        trainY = train[:, -1]
        testX  =  test[:, :-1]
        testY  =  test[:, -1]

        return trainX, trainY, testX, testY 



class Preprocessor:
    def dropEmptyRows(self, data):
        '''
        Parameters:
            data: the data to have rows removed from
        Return:
            the new data with no empty rows
        ''' 
        rows, cols = data.shape
        newData = []
        removedRows = 0

        print("\nRemoving empty rows of data...")
        for row in range(rows):
            if not np.all(data[row, :-1] == 0):
                newData.append(data[row]) 
            else:
                removedRows += 1
                
        print(f"{removedRows} rows of data were dropped.")

        return np.array(newData)
    
    def dropEmptyCols(self, data):
        '''
        Parameters:
            data: the data to have columns removed from
        Return:
            the new data with no empty columns
        ''' 
        rows, cols = data.shape
        newData = []
        removedCols = 0

        print("\nRemoving empty columns of data...")
        for col in range(cols):
            if not np.all(data[:, col] == 0):
                newData.append(data[:, col])
            else:
                removedCols += 1
        print(f"{removedCols} columns of data were dropped.")

        return np.transpose(np.array(newData))
    
    def dropLeastCorrelatedCols(self, headers, data, percentile=30):
        '''
        Parameters:
            headers: the headers of the data
            data: the data to have columns removed from
            percentile: what percent of the data should be removed
        Return:
            the new data with no correlated columns
        ''' 
        # Find the threshold of bottom 30% correlation
        # Use absolute value so we remove columns with small correlations, instead of negative ones
        correlations = np.abs(np.corrcoef(np.transpose(data))[-1])
        threshold = np.percentile(correlations, 30)
        numRemoved = 0
        
        print(f"\nCorrelation threshold for feature removal: {threshold}")
        print("Removing least correlated features...")
        print("Removed features: ", end="")
        for i in range(len(correlations) - 1, -1, -1):
            if correlations[i] <= threshold:
                data = np.delete(data, i, axis=1)

                print(f"{headers[i]}, ", end="")
                headers = np.delete(headers, i)
                
                numRemoved += 1

        rows, cols = data.shape
        print(f"In total, {numRemoved} columns (bottom 30%) were dropped from the array, leaving {cols} columns.")

        return headers, data
    
    def dropLowestBetas(self, headers, data, alpha, percentile=30):
        '''
        Parameters:
            headers: the headers of the data
            data: the data to have columns removed from
            alpha: the alpha value to use for lasso
            percentile: what percent of the data should be removed
        Return:
            the new data with lowest beta columns removed
        '''
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(data[:, :-1], data[:, -1])
        threshold = np.percentile(lasso.coef_, percentile)
        numRemoved = 0

        print(f"\nBeta threshold for feature removal: {threshold}")
        print("Removing features with the lowest beta...")
        print("Removed features: ", end="")
        for i in range(len(lasso.coef_) - 1, -1, -1):
            if lasso.coef_[i] <= threshold:
                data = np.delete(data, i, axis=1)

                print(f"{headers[i]}, ", end="")
                headers = np.delete(headers, i)
                
                numRemoved += 1
        
        rows, cols = data.shape
        print(f"In total, {numRemoved} columns (top 30%) were dropped from the array, leaving {cols} columns.")

        return headers, data



class Evaluator:
    # Print out the accuracy, recall, precision, specificity, and fstat of a model's
    def getMetrics(self, true, predicted, printMetrics=True):
        '''
        Parameters:
            true: the real data
            feature: what a model predicted
        Return:
            A dictionary containing the metrics
        Function:
            Provide information about a model's performance
        '''
        tn, fp, fn, tp = confusion_matrix(true, predicted).ravel()

        # Calculate the metrics
        accuracy = (tp+tn) / (tp+tn+fp+fn)
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        specificity = tn / (tn+fp)
        fstat = 2 * ((precision*recall) / (precision+recall))

        # Print the metrics
        if printMetrics:
            print(f"True Positives: {tp}, False Positives: {fp}, True Negatives: {tn}, False Negatives: {fn}")
            print(f"Accuracy: {accuracy}")
            print(f"Recall: {recall}")
            print(f"Precision: {precision}")
            print(f"Specificity: {specificity}")
            print(f"F-Stat: {fstat}")

        # Return metrics as a dictionary
        metrics = {
            "True Positives": tp,
            "False Positives": fp,
            "True Negatives": tn,
            "False Negatives": fn,
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "Specificity": specificity,
            "FStat": fstat
        }

        return metrics
    

class Tester:
    # testing purposes for conv2d
    def example_conv2d():
        """
        Demonstrates the use of a 2D convolutional layer (Conv2D) in a neural network.
        This function creates a random 4D input tensor with shape (4, 10, 10, 128),
        applies a Conv2D layer with 32 filters, a kernel size of 3x3, and ReLU activation,
        and prints the shape of the resulting output tensor.
        Note:
            - The function assumes that the necessary libraries (e.g., numpy and TensorFlow/Keras)
              are imported and available in the environment.
            - This is an example function and does not return any value.
        Example:
            >>> example_conv2d()
            (4, 8, 8, 32)
        """
        x = np.random.rand(4, 10, 10, 128)
        y = Conv2D(32, 3, activation='relu')(x)
        print(y.shape)