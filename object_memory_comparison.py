import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.spatial.distance import cdist


data_arrays = []
for i in range(1, 7):
    path = f'/Users/giuliadangelo/Downloads/npc-av-learning/core50cropped/workingmemory/s{i}/'
    filenames = glob.glob(path + 'object*memory.npy')
    data = np.array([np.load(f) for f in filenames]).squeeze()
    data_arrays.append(np.copy(data))

    print(f"Shape of data for session {i}: {data.shape}")


# for i in range(len(data_arrays)):
#     for j in range(i, len(data_arrays)):
#         cmat = 1- cdist(data_arrays[i], data_arrays[j], 'cosine')
#         plt.imshow(cmat, interpolation='none')
#         plt.colorbar()
#         plt.title(f'Cosine Similarity Matrix for Session {i+1} vs Session {j+1}')
#         plt.show()



from scipy.special import softmax
summary_data = np.mean(data_arrays, axis=0)
for i in range(len(data_arrays)):
    cmat = 1 - cdist(summary_data, data_arrays[i], 'cosine')
    plt.imshow(softmax(10*cmat,axis=1), interpolation='none')
    plt.colorbar()
    plt.title(f'Cosine Similarity Matrix for Averaged Sessions vs Session {i+1}')
    plt.show()
