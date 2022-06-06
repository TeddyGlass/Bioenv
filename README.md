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

### 3-5. Prediction using machine learning
 We evaluated predictive performances of five machine leanring algorithms, i.e., lightgbm (LGB), xgboost (XGB), random forest (RF), support vector machines (SVM), and neural networkswas (NNs) to evaluate goodness of each protein representations. Endpoints of the prediction were protein localization classification (membrane-binding protein or soluble one) and PPI classification. Details of the verified pairs of algorithms and input features were shown in following table.
|  **ID**  | **Algorithm**  | **Feature** | 
| :---- | :----: | :----: |
|  1   | LGB | BERT |
|  2   | XGB | BERT |
|  3   | RF | BERT |
|  4   | SVM | BERT |
|  5   | NNs | BERT |
|  6   | LGB | Albert |
|  7   | XGB | Albert |
|  8   | RF | Albert |
|  9   | SVM | Albert |
|  10   | NNs | Albert |
|  11   | LGB | T5 |
|  12   | XGB | T5 |
|  13   | RF | T5 |
|  14   | SVM | T5 |
|  15   | NNs | T5 |
|  16   | LGB | T5FT |
|  17   | XGB | T5FT |
|  18   | RF | T5FT |
|  19   | SVM | T5FT |
|  20   | NNs | T5FT |
|  21   | LGB | AAindex |
|  22   | XGB | AAindex |
|  23   | RF | AAindex |
|  24   | SVM | AAindex |
|  25   | NNs | AAindex |
|  26   | LGB | Autcorrelation |
|  27   | XGB | Autcorrelation |
|  28   | RF | Autcorrelation |
|  29   | SVM | Autcorrelation |
|  30   | NNs | Autcorrelation |

### 3-6. Nested cross validation
ML models were optimized outer-five-fold and inner-five-fold nested cross validation. External validation was performed by evaluating the predictive performance with an outer 5-fold cross-validation.

### 3-7. Hyper parameter optimization
Hyper parameters of each ML model were optimized by Bayesian optimization implemented by optuna library. We searched for optimal hyper parameters that minimize avarege value of logloss in inner-three-fold cross validation.

### 3-8. Evaluation metrics
2×2 confusion matrix
|    | **Positive_obs**  | **Negative_obs** |
| :---- | :----: | :---: |
|  **Positive_pred**  | $TP$ | $FT$ |
|  **Negative_pred**  | $FN$ | $TN$ |

* The receiver operating characteristic–area under the curve (ROC–AUC) graphs the performance of a classification model at all thresholds. This curve plots two parameters: the sensitivity (SE) and specificity (SP).  
* Accuracy is defined as the ratio of corrective prediction for all samples that dose not distinguish $TP$ and $TN$.
$$ACC = \frac{TP + TN}{TP + TN + FT + FN }$$
* The SE is defined as the prediction accuracy when the true outcome is positive:  
 $$SE = \frac{TP}{TP + FN}$$
* The SP is defined as the prediction accuracy when the true outcome is negative:  
$$SP = \frac{TN}{TN + FP}$$
* The balanced accuracy (BAC) is the average between the SE and SP:  
$$BAC = \frac{1}{2} (SE + SP)$$
* The Matthews correlation coefficient (MCC) measures the classification accuracy of models for an unbalanced dataset (Matthews, 1975):
$${\displaystyle {\text{MCC}}={\frac {{\mathit {TP}}\times {\mathit {TN}}-{\mathit {FP}}\times {\mathit {FN}}}{\sqrt {({\mathit {TP}}+{\mathit {FP}})({\mathit {TP}}+{\mathit {FN}})({\mathit {TN}}+{\mathit {FP}})({\mathit {TN}}+{\mathit {FN}})}}}}$$


## 4. Results
### 4-1. Visualization of protein representations
Dimensionally compressed protein representations were shown in Fig1-4. Fig.1-2 contains 10 kind of protein localization information, compressed with PCA (Fig. 1) or UMAP (Fig. 2). Fig.3-4 shows tow kind of protein localization, i.e., soluble or membrane-binding, compressed with PCA (Fig. 3) or UMAP (Fig. 4).  


**Fig.1**
![pca_all](/results/DeepLocAll_PCA.png)

**Fig.2**
![ummap_all](/results/DeepLocAll_UMAP.png)

**Fig.3**
![pca_ms](/results/DeepLocMS_PCA.png)

**Fig.4**
![umap_ms](/results/DeepLocMS_UMAP.png)

### 4-2. Machine learning performance
These results were based on a single fold of external validation.
4-2-1. LM 
|  **ID**  | **Algorithm**  | **Feature** | **AUC** | **ACC** | **SE** | **SP** | **BAC** | **MCC** | **cutoff** |
| :---- | :----: | :----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
|  1   | LGB | BERT | 0.916597956	| 0.864529473 | 0.821882952	| 0.893728223	| 0.857805587	| 0.718339	 | 0.317997579 |
|  3   | RF | BERT | 0.885593709	| 0.825232678	| 0.659033079	| 0.93902439	| 0.799028735	| 0.638577986	| 0.472167014 |
|  4   | SVM | BERT | 0.839266874	| 0.815925543	| 0.786259542	| 0.836236934	| 0.811248238	| 0.620150422	| 0.393978849 |
|  5   | NNs | BERT | 0.9158000195	| 0.8490175801	| 0.8218829517	| 0.8675958188	| 0.8447393852	| 0.6878797174	| 0.3519498706	|
|  6   | LGB | Albert | 0.920991037	| 0.844881075	| 0.809160305	| 0.869337979	| 0.839249142	| 0.678498284	| 0.291703777 |
|  8   | RF | Albert | 0.859625325	| 0.804550155	| 0.648854962	| 0.911149826 | 0.780002394	| 0.591401632	| 0.46296315 |
|  9   | SVM | Albert | 0.494855529	| 0.593588418	| 0	| 1	| 0.5	| 0	| 1.406519034 |
|  10   | NNs | Albert | 0.9257387558	| 0.8645294726	| 0.8320610687	| 0.8867595819	| 0.8594103253	| 0.7191107991	| 0.3182281256 |
|  11   | LGB | T5 | 0.936200583	| 0.855222337	| 0.86259542	| 0.850174216	| 0.856384818	| 0.705354276	| 0.276454328 |
|  13   | RF | T5 | 0.903219228	| 0.807652534	| 0.839694656	| 0.785714286	| 0.81270447	| 0.615581625	| 0.349313724 |
|  14   | SVM | T5 | 0.864701084	| 0.830403309	| 0.82697201	| 0.832752613	| 0.829862312	| 0.653548084	| 0.393951783|

4-2-2 Descriptors
|  **ID**  | **Algorithm**  | **Feature** | **AUC** | **ACC** | **SE** | **SP** | **BAC** | **MCC** | **cutoff** |
| :---- | :----: | :----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
|  21   | LGB | AAindex | 0.8147325584	| 0.7652533609	| 0.6335877863	| 0.8554006969	| 0.7444942416	| 0.5058187773	| 0.4141976214
|  23   | RF | AAindex | 0.7819639865	| 0.7435367115	| 0.5954198473	| 0.8449477352	| 0.7201837913	| 0.4585887195	| 0.4248247629
|  24   | SVM | AAindex | 0.7505873696	| 0.728024819	| 0.5547073791	| 0.8466898955	| 0.7006986373	| 0.4239017386	| 0.423105938
|  25   | NNs | AAindex | 0.8056892837	| 0.74663909	| 0.6234096692	| 0.831010453	| 0.7272100611	| 0.4668278033	| 0.4505890608
