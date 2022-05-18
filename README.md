# Comparison of Protein Representation

## 1. Enviornment
* Miniconda virtual enviornment  
* Enviornment name: ```bioenv_ver0.1``` 
* Python 3.9.12

## 2. Installation of packages
Try following script
```bash
bash installation_packages.sh
```
To install BioMed, please try following scripts

```bash
git clone https://github.com/gadsbyfly/PyBioMed
```

```bash
cd ./PyBioMed
```
```bash
python setup.py install
```
Installation details can be available in [GitHub](https://github.com/gadsbyfly/PyBioMed).

## 3. Materials and Methods

### 3-1. Benchmark deta set
1) **DeepLoc**  
We used DeepLoc database that contains protein sequences with the information of their localization, which was used in previous study conducted by [Elnaggar *et al.* (2021, *IEEE Trans. Pattern Anal.*)](https://github.com/agemagician/ProtTrans). This dataset can be used not only for classification task of predicting whether it is membrane-binding protein or soluble one but also for visualization of various protein localization.

2) **DeepPPI**  
DeepPPI data set which was used for training of DeepPPI ([Du *et al.*, **2017**, *J. Chem. Inf. Model.*](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00028)) contains protein-protein interaction on *Saccharomyces cerevisiae* and thier UniProt protein IDs. DeepPPI data set can be used for classification task that predicting PPI. However, each FASTA file or sequence must be downloaded from UniProt, separately.

|  **Database**  |  **size**  |
| :---- | ----: |
| DeepLoc | 8,464 (4832 'M' or 'S')|
| DeepPPI | 65,851|

### 3-2. Calculation of protein representation
**Language model**
|  **Methods**  |  **Note**  | **Dimension** | 
| :---- | :----: | ---: |
|  Albert_BFD   |  Albert pre-trained on  BFD | 4,096 |
|  BERT_BFD   |  BERT pre-trained on BFD | 1,024 |
|  T5_BFD   |  T5 pre-trained on BFD | 1,024 |
|  T5_FT   |  T5 trained on BFD and finetuned on UniRef50  | 1,024 |

**Sequence descriptor**
|  **Methods**  |  **Note**  | **Dimension** | 
| :---- | :----: | ---: |
|  AAindex1   | Physicochemical property | 566 |
|  Autocorrelation   |  Topology  | 720 |

### 3-3. Tools for calculating protein representation
1. **Bio-embeddings**  
Bio-embeddings is a Python package to calculate sequence embedding for amino acids. This package provides us with general embedding methods from binary embedding to state of the art language model such as Word2Vec, LSTM, and Transformer.  
2. **PyBioMed**  
[PyBioMed](https://github.com/gadsbyfly/PyBioMed) is a Python package to calculate not only protein descriptors but also chemical and DNA descriptors ([Dong *et al.*, **2018**, *J. Cheminform.*](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0270-2)). 

### 3-4. Data visualization
We attempted data visualization of the DeepLoc data set using PCA and UMAP algorithms to investigate whether protein localization is separated by their numerical representation.