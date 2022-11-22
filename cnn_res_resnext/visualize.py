"""
Created on Wed Oct 19 00:29:38 2022
Visualizing Features before FC layer

"""
import models
import torch
import os
import numpy as np
import data
import torch.nn as nn
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name == "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name == "layer4": x = F.avg_pool2d(x, 14)
            if name in self.extracted_layers:
                outputs.append(x.detach().to('cpu').numpy())
        return outputs
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='hw1')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='dataset path')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/', help='model save path')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    input_size = 224
    batch_size = 24
    ptpath = '.\\checkpoints\\best_model_res18.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = data.load_data(data_dir=data_dir,input_size=input_size, batch_size=batch_size)
    
    num_classes = 10
    model = models.model_B(num_classes)
    model.load_state_dict(torch.load(ptpath))
    model = model.to(device)
              
    extract_result = FeatureExtractor(model,"fc")

    features = np.empty([24,10])
    lab = np.empty([24,])
    for inputs,labels in train_loader:
        inputs = inputs.to(device)
        fea = extract_result(inputs)
        features = np.append(features,fea[0],axis=0)
        lab = np.append(lab,labels,axis=0)

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    X = np.nan_to_num(features)
    y = lab
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), 
                  fontdict={'weight': 'bold', 'size': 4})
    plt.scatter(X_norm[:,0],X_norm[:,1],c=y, s=1, alpha =1)
    plt.xticks([])
    plt.yticks([])
    plt.show()
