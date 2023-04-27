from sklearn.model_selection import KFold
from glob import glob
import pandas as pd

kfold = KFold(n_splits=5, shuffle=True)

for channel in ['A', 'P', 'R']:

    #a =glob(f"/Users/beppe2hd/Data/Microplastiche/HMPD-Gen/images/*_{channel}.bmp")
    df = pd.read_csv("/Users/beppe2hd/Data/Microplastiche/HMPD-Gen/gt.csv")#  pd.DataFrame(a)

    for i, indexes in enumerate(kfold.split(df)):
        dfTrain = df.iloc[indexes[0]]
        dfTest = df.iloc[indexes[1]]
        dfTrain.to_csv(f"./{channel}/train_{i}.csv", index=False, header=False)
        dfTest.to_csv(f"./{channel}/test_{i}.csv", index=False, header=False)

