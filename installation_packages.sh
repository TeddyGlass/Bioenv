# create new envirnmenr
# conda create -n bioenv_ver0.1 python==3.8.5

# install packages
conda install numpy pandas matplotlib -y
conda install -c anaconda jupyter seaborn scikit-learn -y
conda install -c conda-forge xgboost -y
conda install -c conda-forge lightgbm -y 
conda install -c conda-forge tensorflow -y
conda install -c conda-forge keras -y
conda install -c conda-forge optuna -y
conda install -c conda-forge umap-learn -y
conda install -c conda-forge biopython -y
conda install -c conda-forge rdkit -y
pip install 'bio-embeddings[all]'
pip install ilearnplus

# install PyBioMed
git clone https://github.com/gadsbyfly/PyBioMed
cd ./PyBioMed
python setup.py install
