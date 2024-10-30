import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

filepath = "/Users/kayal/Documents/Southampton/3rd Year/Sem 1/BSc Physics Project - ML for Lattice QCD/CorrelationHeatmap/2pt-3pt-qsqmax-scalar.gpl"

def processData(data, knownStrings, fromFile=False, fileName=None):
    arrays = {label: [] for label in knownStrings}  # Dictionary to store multiple arrays for each label
    currentArray = []
    currentString = None
    skipFirst = True  # Flag to skip the first float in every array

    if fromFile and fileName:
        with open(fileName, 'r') as file:
            data = file.read().split()

    for value in data:
        if value in knownStrings:
            # If we're in the middle of an array for the previous string, append it to the list
            if currentString and currentArray:
                arrays[currentString].append(np.array(currentArray, dtype=float))
                currentArray = []
            currentString = value
            skipFirst = True  # Reset flag to skip the first float
        else:
            try:
                if skipFirst:
                    skipFirst = False  # Skip the first float in this array
                else:
                    currentArray.append(float(value))
            except ValueError:
                pass

    # Add the last array to its corresponding string label
    if currentString and currentArray:
        arrays[currentString].append(np.array(currentArray, dtype=float))

    return arrays

# Identifiers to sort data
knownStrings = [
    '2pt_D_gold_msml5_fine.ll', 
    '2pt_D_nongold_msml5_fine.ll', 
    '2pt_msml5_fine_K_zeromom.ll',
    'localtempvec_pmax_3pt_T16_msml5_fine.ll', 
    'localtempvec_pmax_3pt_T19_msml5_fine.ll',
    'localtempvec_pmax_3pt_T22_msml5_fine.ll', 
    'localtempvec_pmax_3pt_T25_msml5_fine.ll'
]

# Create arrays from file
arraysFromFile = processData([], knownStrings, fromFile=True, fileName=filepath)

# Function creating time step arrays for each identifier
def extractNthValuesForLabel(arrays, label, maxTime=len(arraysFromFile['2pt_D_gold_msml5_fine.ll'][6])):
    # Check if the label exists in the dictionary
    if label not in arrays:
        print(f"Label {label} not found.")
        return None
    
    # Dictionary to hold arrays for each time step for the specific label
    timeArrays = {time: [] for time in range(maxTime)}
    
    # Loop over time (from 0 to maxTime - 1)
    for time in range(maxTime):
        for array in arrays[label]:
            if len(array) > time:  # Ensure the array has enough values
                timeArrays[time].append(array[time])  # Extract the nth value

    return timeArrays

# Extract nth values for a specific label and group them by time

timeArrays2DG = extractNthValuesForLabel(arraysFromFile, knownStrings[0], maxTime=16)
timeArrays2DNG = extractNthValuesForLabel(arraysFromFile, knownStrings[1], maxTime=16)
timeArrays2K = extractNthValuesForLabel(arraysFromFile, knownStrings[2], maxTime=16)
timeArrays3T16 = extractNthValuesForLabel(arraysFromFile, knownStrings[3], maxTime=16)
timeArrays3T19 = extractNthValuesForLabel(arraysFromFile, knownStrings[4], maxTime=16)
timeArrays3T22 = extractNthValuesForLabel(arraysFromFile, knownStrings[5], maxTime=16)
timeArrays3T25 = extractNthValuesForLabel(arraysFromFile, knownStrings[6], maxTime=16)

# Function to calculate correlations for every time step of a time array vs time step of another time array
def calculateTimeStepCorrelations(timeArray1, timeArray2, maxTime=16, capLength=400):
    
    correlationMatrix = np.zeros((maxTime, maxTime))
    
    # Iterate over each time step in both time arrays
    for i in range(maxTime):
        for j in range(maxTime):
            # Cap each time step array to the first 'capLength' elements (as not all identifiers have the same no. elements)
            values1 = timeArray1[i][:capLength]
            values2 = timeArray2[j][:capLength]
            
            # Ensure both arrays have the same length and more than 1 value
            if len(values1) == len(values2) and len(values1) > 1:
                # Calculate the Pearson correlation coefficient
                corr, _ = pearsonr(values1, values2)
                correlationMatrix[i, j] = corr
            else:
                # If lengths don't match or not enough data, set correlation to NaN
                correlationMatrix[i, j] = np.nan

    return correlationMatrix

# List of time arrays for convenience

twoPtArraysList = [
    timeArrays2DG,
    timeArrays2DNG,
    timeArrays2K
]

threePtArraysList = [
    timeArrays3T16,
    timeArrays3T19,
    timeArrays3T22,
    timeArrays3T25
]

# Choose which time arrays to use for heatmap
twoPtArray = twoPtArraysList[1]
threePtArray = threePtArraysList[2]

pwCorrelationMatrix = calculateTimeStepCorrelations(twoPtArray, threePtArray, maxTime=16)

# Print the resulting correlation matrix
print("Time-Step Correlation Matrix:")
print(pwCorrelationMatrix)

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pwCorrelationMatrix, annot=False, cmap="cool", cbar=True, 
            xticklabels=np.arange(1, 16), yticklabels=np.arange(1, 16))

# Function to convert array name to string for heatmap labels
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]

twoPtArrayName=namestr(twoPtArray, globals())
threePtArrayName=namestr(threePtArray, globals())

# Add labels and title
heatTitle = f'Correlation Matrix between time steps of {twoPtArrayName} and {threePtArrayName}'

plt.title(heatTitle)
plt.xlabel(f"Time Steps of {threePtArrayName}")
plt.ylabel(f"Time Steps of {twoPtArrayName}")
plt.grid(visible=False)


# Show the plot
plt.show()