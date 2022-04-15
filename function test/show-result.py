import numpy as np
from matplotlib import pyplot as plt

X = []
for i in range(1000):
    X.append(i + 1)
# X = np.arange(1, iter_num + 1).astype(dtype=str)
pso_results = np.loadtxt('C:/app/source/iQPSO/function test/pso-result.txt', delimiter='\t')
qpso_results = np.loadtxt('C:/app/source/iQPSO/function test/qpso-result.txt', delimiter='\t')
aqpso_results = np.loadtxt('C:/app/source/iQPSO/function test/aqpso-result.txt', delimiter='\t')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(X, pso_results, label='PSO')
plt.plot(X, qpso_results, label='QPSO')
plt.plot(X, aqpso_results, label='iQPSO')
plt.legend()
plt.xlabel('迭代次数', size=15)
plt.ylabel('适应度', size=15)
plt.show()
