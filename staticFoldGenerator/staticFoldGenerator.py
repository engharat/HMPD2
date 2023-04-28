from sklearn.model_selection import KFold
from glob import glob
import pandas as pd

kfold = KFold(n_splits=5, shuffle=True, random_state=43)

#a =glob(f"/Users/beppe2hd/Data/Microplastiche/HMPD-Gen/images/*_{channel}.bmp")
df = pd.read_csv("/Users/beppe2hd/Data/Microplastiche/HMPD-Gen/gtPossible.csv")#  pd.DataFrame(a)

for i, indexes in enumerate(kfold.split(df)):
        dfTrain = df.iloc[indexes[0]]
        dfTest = df.iloc[indexes[1]]
        dfTrain.to_csv(f"./GT_possible/train_{i}.csv", index=False, header=False)
        dfTest.to_csv(f"./GT_possible/test_{i}.csv", index=False, header=False)

