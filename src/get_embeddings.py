from language_model.embedding import LMEmbedder
import pandas as pd
import numpy as np
import torch


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
    # load DeepLoc data set
    df = pd.read_csv('../data/DeepLoc/DeepLocAll.csv')
    sequences = df.iloc[:,0].tolist()
    sequences = [''.join(seq.split()) for seq in sequences]
    # embedding
    for name in ['Albert_BFD', 'BERT_BFD', 'T5_BFD', 'T5_FT', 'XLNet_Uniref100']:
        print('-'*100)
        if name == 'Albert_BFD':
            embedder = LMEmbedder(device=torch.device('cuda:1'))
            feature = embedder.albert_embedding(sequences)
            print(name, feature.shape)
            pd.concat([df.iloc[:,1:], feature], axis=1).to_csv(f'../data/DeepLoc/DeepLocEmbedd_{name}.csv', index=False)
        if name == 'BERT_BFD':
            embedder = LMEmbedder(device=torch.device('cuda:0'))
            feature = embedder.bert_embedding(sequences)
            print(name, feature.shape)
            pd.concat([df.iloc[:,1:], feature], axis=1).to_csv(f'../data/DeepLoc/DeepLocEmbedd_{name}.csv', index=False)
        if name == 'T5_BFD':
            embedder = LMEmbedder(device=torch.device('cuda:1'))
            feature = embedder.t5_embedding(sequences)
            print(name, feature.shape)
            pd.concat([df.iloc[:,1:], feature], axis=1).to_csv(f'../data/DeepLoc/DeepLocEmbedd_{name}.csv', index=False)
        if name == 'T5_FT':
            embedder = LMEmbedder(device=torch.device('cuda:2'))
            feature = embedder.t5ft_embedding(sequences)
            print(name, feature.shape)
            pd.concat([df.iloc[:,1:], feature], axis=1).to_csv(f'../data/DeepLoc/DeepLocEmbedd_{name}.csv', index=False)
        if name == 'XLNet_Uniref100':
            embedder = LMEmbedder(device=torch.device('cuda:3'))
            feature = embedder.xlnet_embedding(sequences)
            print(name, feature.shape)
            pd.concat([df.iloc[:,1:], feature], axis=1).to_csv(f'../data/DeepLoc/DeepLocEmbedd_{name}.csv', index=False)
        print('-'*100)