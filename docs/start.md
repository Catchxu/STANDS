STANDS is a powerful documentation framework to detect, align and subtyping anomalous tissue domains across multiple samples. In the subsequent sections, we will introduce the preparatory tasks before utilizing STANDS for your research, including the installation of Python packages, downloading of datasets, and other related procedures.




## Installation
STANDS is developed as a Python package. You will need to install Python, and the recommended version is Python 3.9.5.

You can download the package from GitHub and install it locally:

```commandline
git clone https://github.com/Catchxu/STANDS.git
cd STANDS/
python3 setup.py install --user
```




## Datasets
All experimental datasets involved in this paper are available from their respective original sources: the 10x-Visium datasets of healthy human breast tissues (10x-hNB datasets) are available at the [CELLxGENE](https://cellxgene.cziscience.com/collections/4195ab4c-20bd-4cd3-8b3d-65601277e731); The 10x-Visium datasets of human breast cancer tissues (10x-hBC datasets) are available at the [github](https://github.com/almaan/her2st/tree/master); The scRNA-seq dataset of human pancreatic ductal (sc-hPD) and 10x-Visium datasets of the human pancreatic ductal adenocarcinomas (10x-hPDAC) are available at the [Gene Expression Omnibus](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE111672); The slide-seqV2 datasets of mouse embryo tissues (ssq-mEmb datasets) are available at the [CELLxGENE](https://cellxgene.cziscience.com/collections/d74b6979-efba-47cd-990a-9d80ccf29055).

We also provide organized and processed small datasets to demo the our code and tutorials.
You can download the demo datasets from [Google Drive](https://drive.google.com/file/d/1_eaOOiBfJtM-OZ3Ptdkylubn5cD17aUs/view?usp=drive_link).




## Getting help
See the tutorial for more complete documentation of all the functions of STANDS.

For questions or comments, please use the [GitHub issues](https://github.com/Catchxu/STANDS/issues).




## Tested environment
- CPU: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
- CPU Memory: 256 GB
- GPU: NVIDIA GeForce RTX 3090 
- GPU Memory: 24 GB
- System: Ubuntu 20.04.5 LTS
- Python: 3.9.15




## Main dependencies
- torch==1.13.0
- dgl==1.1.1
- torchvision==0.14.1
- anndata==0.10.3
- numpy==1.19.2
- scanpy==1.9.6
- scipy==1.9.3
- sklearn==0.0.post2
- pandas==1.5.2
- squidpy==1.2.2
- setuptools==59.5.0