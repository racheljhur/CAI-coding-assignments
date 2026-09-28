import numpy as np
import matplotlib.pyplot as plt

train_fractions = [0.05, 0.10, 0.25, 1.00]

crf_span_f1 = np.empty((4, 1)) 
bilstm_span_f1 = np.empty((4,1)) 
for i, frac in enumerate(train_fractions):
    crf_span_f1[i] = np.load(f'tokenSpanF1_trainFrac_{frac}.npy')[1] * 100
    bilstm_span_f1[i] = np.load(f'tokenSpanF1_trainFrac_{frac}_bilstm.npy')[1] * 100

plt.figure()
plt.plot(np.array(train_fractions), crf_span_f1, label = 'CRF', color = 'red', marker='o')
plt.plot(np.array(train_fractions), bilstm_span_f1, label = 'BiLSTM', color = 'blue', marker='o')
plt.legend()
plt.xlabel(f'Fraction of Training Set')
plt.ylabel(f'F1 Score (%)')
plt.title('Data Efficiency Curves')
plt.savefig('dataEffPlot.png', dpi = 300)

