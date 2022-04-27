import torch 
from matplotlib import pyplot as plt



ckpt_path = '/media/altex/XcDrive/projects/im2latex-pytorch/ckpt-modelV0_1-last.pt'
ckpt = torch.load(ckpt_path)






plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.plot(ckpt['train_loss'],'-',ckpt['val_loss'],'-')
plt.legend(["train","val"])
plt.title("loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.subplot(2,2,2)
plt.plot(ckpt['train_accuracy'],'-',ckpt['val_accuracy'],'-')
plt.title("accuracy")
plt.legend(["train","val"])
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.grid()
plt.show()