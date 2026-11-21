import pandas as pd
import logging
import os 

def main():
    filepath = "labeled_data"
    csvFilePath = os.path.join(filepath,"EURJPY_short_scaled.csv")
    sampleCSV  = pd.read_csv(csvFilePath)

    partOfSample = sampleCSV[:10]
    #print(partOfSample)
    samplesShape = sampleCSV.shape
    print(samplesShape)
if __name__ == "__main__":
    main()
