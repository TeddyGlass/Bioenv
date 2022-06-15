from language_model.embedding import LMEmbedder
import pandas as pd
import torch
import os
from Bio import SeqIO
from tqdm import tqdm


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
    # load DeepPPI sequences
    fasta_path = '../data/DeepPPI/DeepPPI.fasta'
    seq_dict = {}
    for i, record in enumerate(tqdm(SeqIO.parse(fasta_path, 'fasta'), total=4424)):
        Id = record.id.split('|')[1]
        seq = str(record.seq)
        seq_dict[str(Id)] = seq
    df = pd.DataFrame(seq_dict.keys(), columns = ['UniprotID'])
    sequences = list(seq_dict.values())
    # embedding
    name = 'T5_BFD'
    embedder = LMEmbedder(device=torch.device('cuda:2'))
    feature = embedder.t5_embedding(sequences)
    print(name, feature.shape)
    pd.concat([df, pd.DataFrame(feature)], axis=1).to_csv(f'../data/DeepPPI/DeepPPIEmbedd_{name}.csv', index=False)
    print('-'*100)