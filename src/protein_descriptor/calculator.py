from PyBioMed.PyProtein import AAIndex, Autocorrelation
import os
from tqdm import tqdm
import pandas as pd
import numpy as np


def calc_aaindex(sequences: list):
    # initialize
    if not os.path.exists('aaindex1'):
        AAIndex.GetAAIndex1('ANDN920101')
    # get all AAindex name
    f = open('aaindex1')
    data = f.read()
    idxes = []
    for i, item in  enumerate(data.split('//')):
        if i == len(data.split('//')) -1:
            break
        else:
            idxes.append(item.split()[1])
    # create AAindex1 dictionary
    print('-'*100)
    print('crating all AAindex1 dictionary ...')
    AAindex1 = {}
    for idx in tqdm(idxes):
        AAindex1[idx] = AAIndex.GetAAIndex1(idx)
    print('completed')
    print('-'*100)
    # calculate AAindex1 descriptor
    print('-'*100)
    print('calculating AAindex1 descriptors ...')
    descriptor_matrix = np.zeros([len(sequences), len(AAindex1)])
    for i, seq in enumerate(tqdm(sequences, total=len(sequences))):
        for j, aaindex1_dict in enumerate(AAindex1.values()):
            ss = []
            for s in seq:
                try:
                    if aaindex1_dict[s] is not None:
                        ss.append(aaindex1_dict[s])
                except KeyError:
                    pass
            descriptor_matrix[i,j] = np.mean(ss)
    print('completed')
    print('-'*100)
    return pd.DataFrame(descriptor_matrix, columns = list(AAindex1.keys()))


def calc_autocorr(sequences: list):
    descriptor_matrix = np.full((len(sequences), 720), np.nan)
    columns = Autocorrelation.CalculateAutoTotal('ALANINE').keys()
    print('-'*100)
    print('calculating Autocorrelation descriptors ...')
    for i, seq in enumerate(tqdm(sequences, total=len(sequences))):
        try: 
            descriptor = Autocorrelation.CalculateAutoTotal(seq)
            descriptor_matrix[i] = np.array(list(descriptor.values()))
        except KeyError:
            pass
    print('completed')
    print('-'*100)
    return pd.DataFrame(descriptor_matrix, columns=columns)