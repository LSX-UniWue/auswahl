import pandas as pd
import numpy as np
from typing import List


def build_hierarchical_index(structure: List[List[str]]):

    index = np.array([structure[-1]])
    for i in range(len(structure) - 2, -1, -1):
        index = np.concatenate([
            np.repeat([structure[i]], repeats=index.shape[1], axis=1),
            np.tile(index, (1, len(structure[i])))
        ])
    return index.tolist()

index = build_hierarchical_index([
    ['CARS', 'VIP'],
    ['n=1', 'n=2'],
    ['mse', 'mae'],
    ['mean', 'std', *[f'sample_{i}' for i in range(2)]]
])

#print(index.shape)
#data = np.random.rand(index.shape[1], 1)
#print(data.shape)
#frame = pd.DataFrame(data, index=index)

# maintain a regression frame and a feature frame
# ----> should work...

#--> theoretically, the methods itself can be handled as rows

df = pd.DataFrame(np.random.randn(4, 32), columns=index)
print(df["VIP", :, 'mse', 'mean'])

# this is simply perfect


