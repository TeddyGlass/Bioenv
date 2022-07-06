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

### 3-6. Allocation of data set
20% of total data was used for the test set, and the other 80% of total data was used for the validation set. ML models were optimized five-fold cross validation. 

### 3-7. Hyper parameter optimization
Hyper parameters of each ML model were optimized by Bayesian optimization implemented by optuna library. We searched for optimal hyper parameters that minimize avarege value of logloss in five-fold cross validation.

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
![pca_all](/results/20220703_DeepLocAll_PCA.png)

**Fig.2**
![ummap_all](/results/20220703_DeepLocAll_UMAP.png)

**Fig.3**
![pca_ms](/results/20220706_DeepLocMS_PCA.png)

**Fig.4**
![umap_ms](/results/20220706_DeepLocMS_UMAP.png)

### 4-2. Machine learning performances in localization prediction  
4-2-1. ML performances using LM features 
ID | Algorithm | Feature | auc | acc | sen | spe | bac | mcc | cutoff
-- | -- | -- | -- | -- | -- | -- | -- | -- | --
1 | LGB | BERT | 0.9165979555 | 0.8645294726 | 0.8218829517 | 0.893728223 | 0.8578055873 | 0.7183389996 | 0.3179975791
3 | RF | BERT | 0.8855937087 | 0.8252326784 | 0.6590330789 | 0.9390243902 | 0.7990287346 | 0.6385779858 | 0.4721670139
4 | SVM | BERT | 0.8392668741 | 0.8159255429 | 0.786259542 | 0.8362369338 | 0.8112482379 | 0.6201504217 | 0.3939788489
5 | NNs | BERT | 0.9158000195 | 0.8490175801 | 0.8218829517 | 0.8675958188 | 0.8447393852 | 0.6878797174 | 0.3519498706
6 | LGB | Albert | 0.9209910365 | 0.8448810755 | 0.8091603053 | 0.8693379791 | 0.8392491422 | 0.6784982844 | 0.2917037768
8 | RF | Albert | 0.8596253247 | 0.8045501551 | 0.6488549618 | 0.9111498258 | 0.7800023938 | 0.5914016315 | 0.4629631502
9 | SVM | Albert | 0.4948555293 | 0.5935884178 | 0 | 1 | 0.5 | 0 | 1.406519034
10 | NNs | Albert | 0.9257387558 | 0.8645294726 | 0.8320610687 | 0.8867595819 | 0.8594103253 | 0.7191107991 | 0.3182281256
11 | LGB | T5 | 0.9362005834 | 0.8552223371 | 0.8625954198 | 0.850174216 | 0.8563848179 | 0.705354276 | 0.2764543277
13 | RF | T5 | 0.9032192285 | 0.8076525336 | 0.8396946565 | 0.7857142857 | 0.8127044711 | 0.6155816246 | 0.3493137237
14 | SVM | T5 | 0.8647010843 | 0.8304033092 | 0.8269720102 | 0.8327526132 | 0.8298623117 | 0.6535480844 | 0.3939517833
15 | NNs | T5 | 0.9419767535 | 0.8738366081 | 0.8727735369 | 0.8745644599 | 0.8736689984 | 0.7416102058 | 0.302672714
16 | LGB | T5FT | 0.9403099538 | 0.8728024819 | 0.8371501272 | 0.8972125436 | 0.8671813354 | 0.7358811347 | 0.2815121038
18 | RF | T5FT | 0.9082063285 | 0.8252326784 | 0.8549618321 | 0.8048780488 | 0.8299199404 | 0.6499322633 | 0.3661924918
19 | SVM | T5FT | 0.868059065 | 0.8304033092 | 0.8473282443 | 0.818815331 | 0.8330717876 | 0.6574177289 | 0.3743409225
20 | NNs | T5FT | 0.9379338777 | 0.8769389866 | 0.7989821883 | 0.9303135889 | 0.8646478886 | 0.7435737539 | 0.4507674873

4-2-2. ML performances using Descriptors
ID | Algorithm | Feature | auc | acc | sen | spe | bac | mcc | cutoff
-- | -- | -- | -- | -- | -- | -- | -- | -- | --
21 | LGB | AAindex | 0.8147325584 | 0.7652533609 | 0.6335877863 | 0.8554006969 | 0.7444942416 | 0.5058187773 | 0.4141976214
23 | RF | AAindex | 0.7819639865 | 0.7435367115 | 0.5954198473 | 0.8449477352 | 0.7201837913 | 0.4585887195 | 0.4248247629
24 | SVM | AAindex | 0.7505873696 | 0.728024819 | 0.5547073791 | 0.8466898955 | 0.7006986373 | 0.4239017386 | 0.423105938
25 | NNs | AAindex | 0.8056892837 | 0.74663909 | 0.6234096692 | 0.831010453 | 0.7272100611 | 0.4668278033 | 0.4505890608
26 | LGB | Autcorrelation | 0.7943098297 | 0.74663909 | 0.582697201 | 0.8588850174 | 0.7207911092 | 0.464729527 | 0.4527447606
28 | RF | Autcorrelation | 0.7149683929 | 0.6690796277 | 0.6030534351 | 0.7142857143 | 0.6586695747 | 0.3163701299 | 0.4007631177
29 | SVM | Autcorrelation | 0.4578490305 | 0.4653567735 | 0.5623409669 | 0.3989547038 | 0.4806478354 | -0.0385861426 | 0.4061854939
30 | NNs | Autcorrelation | 0.7292071176 | 0.6980351603 | 0.572519084 | 0.7839721254 | 0.6782456047 | 0.3645810535 | 0.483382225

### 4-3. Machine learning performances in PPI prediction  
4-3-1. ML performances using LM features 
ID | Algorithm | Feature | auc | acc | sen | spe | bac | mcc | cutoff
-- | -- | -- | -- | -- | -- | -- | -- | -- | --
1 | LGB | BERT | 0.9809597845 | 0.942753018 | 0.926419467 | 0.948554378 | 0.9374869225 | 0.8562765166 | 0.2449948071
3 | RF | BERT | 0.9341718542 | 0.8962113735 | 0.8432792584 | 0.9150118325 | 0.8791455454 | 0.7397290142 | 0.2608350544
5 | NNs | BERT | 0.9488463751 | 0.8947688103 | 0.8754345307 | 0.9016359708 | 0.8885352507 | 0.7442958339 | 0.2293150723
6 | LGB | Albert | 0.9804649707 | 0.9395641941 | 0.9298957126 | 0.9429982508 | 0.9364469817 | 0.8496837163 | 0.2368388973
8 | RF | Albert | 0.8900949681 | 0.8533900235 | 0.7841830823 | 0.8779709847 | 0.8310770335 | 0.6381692139 | 0.2722957675
10 | NNs | Albert | 0.9682535356 | 0.9214941918 | 0.907589803 | 0.9264327606 | 0.9170112818 | 0.8065450144 | 0.2274309397
11 | LGB | T5 | 0.9815264017 | 0.9445752031 | 0.9313441483 | 0.9492746167 | 0.9403093825 | 0.861092561 | 0.24288333
13 | RF | T5 | 0.9302965324 | 0.8790524637 | 0.8609501738 | 0.8854820455 | 0.8732161096 | 0.7096662719 | 0.1866970443
15 | NNs | T5 | 0.9647514032 | 0.9145850733 | 0.9003476246 | 0.9196419385 | 0.9099947815 | 0.7905234469 | 0.2409929633
16 | LGB | T5FT | 0.9818174302 | 0.9404752866 | 0.9342410197 | 0.9426895771 | 0.9384652984 | 0.8523951619 | 0.218906376
18 | RF | T5FT | 0.9396301721 | 0.8978057854 | 0.8525492468 | 0.9138800288 | 0.8832146378 | 0.7451032363 | 0.2371564418
20 | NNs | T5FT | 0.9578550371 | 0.8974261635 | 0.8945538818 | 0.8984463422 | 0.896500112 | 0.7544499989 | 0.211178124

4-3-2 ML performances using Descriptors
ID | Algorithm | Feature | auc | acc | sen | spe | bac | mcc | cutoff
-- | -- | -- | -- | -- | -- | -- | -- | -- | --
21 | LGB | AAindex | 0.9772303048 | 0.9346291094 | 0.9241019699 | 0.9383681449 | 0.9312350574 | 0.8378769112 | 0.2362097712
23 | RF | AAindex | 0.9185898666 | 0.854680738 | 0.8568945539 | 0.8538944336 | 0.8553944937 | 0.6635603994 | 0.1976461332
25 | NNs | AAindex | 0.8825347419 | 0.811251993 | 0.7914252607 | 0.8182940632 | 0.8048596619 | 0.5654337732 | 0.2719802558
26 | LGB | Autcorrelation | 0.9777245226 | 0.9315921342 | 0.9269988413 | 0.9332235827 | 0.930111212 | 0.8318182397 | 0.2241877723
28 | RF | Autcorrelation | 0.8026669488 | 0.7689621137 | 0.6938006952 | 0.7956579895 | 0.7447293424 | 0.4568579212 | 0.2805243307
30 | NNs | Autcorrelation | 0.9411293828 | 0.8778376737 | 0.8707995365 | 0.8803374833 | 0.8755685099 | 0.7101987432 | 0.2113927603
