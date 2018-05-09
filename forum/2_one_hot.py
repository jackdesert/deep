import pdb
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return(results)

sequences = [[3, 5], [4, 2]]
bb = vectorize_sequences(sequences, 10)
pdb.set_trace()
