from language_model.embedding import LMEmbedder
import pandas as pd
import torch
import os

def get_gpu_properties():
    print('-'*100)
    print('GPU availability:', torch.cuda.is_available())
    print('-'*100)
    print('Available GPU counts:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
    print('-'*100)
    print('Current device:', torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
    print('-'*100)


if __name__ == '__main__':
    get_gpu_properties()
    # load DeepLoc data set
    df = pd.read_csv('../data/DeepLoc/DeepLocAll.csv')
    sequences = df.iloc[:,0].tolist()
    sequences = [''.join(seq.split()) for seq in sequences]
    # embedding
    name = 'XLNet_Uniref100'
    embedder = LMEmbedder(device=torch.device('cuda:0'))
    feature = embedder.xlnet_embedding(sequences)
    print(name, feature.shape)
    pd.concat([df.iloc[:,1:], feature], axis=1).to_csv(f'../data/DeepLoc/DeepLocEmbedd_{name}.csv',  ndex=False)
    print('-'*100)