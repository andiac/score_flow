import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
import matplotlib.pyplot as plt

ind_file_path = "./cifar10_cifar10_vp_bpd.txt" 
ood_file_path = "./svhn_cifar10_vp_bpd.txt" 

with open(ind_file_path, "r") as f:
  ind_bpds = np.array(list(map(float, f.readlines())))

with open(ood_file_path, "r") as f:
  ood_bpds = np.array(list(map(float, f.readlines())))

ind_bpds = -ind_bpds
ood_bpds = -ood_bpds
print(roc_auc_score(np.concatenate((np.ones(10000), np.zeros(10000))), np.concatenate((ind_bpds[:10000], ood_bpds[:10000]))))

plt.hist(ind_bpds[:10000], bins=200)
plt.hist(ood_bpds[:10000], bins=200)
plt.savefig("hist.png")
