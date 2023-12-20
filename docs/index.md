# Detecting and dissecting anomalous anatomic regions in spatial transcriptomics with STANDS
We introduce <b>S</b>patial <b>T</b>ranscriptomics <b>AN</b>omaly <b>D</b>etection and <b>S</b>ubtyping (**STANDS**), an innovative computational method capable of integrating multimodal information, 
e.g., spatial gene expression, histology image and single cell gene expression, to not only delineate anomalous tissue regions but also reveal 
their compositional heterogeneities across multi-sample spatial transcriptomics (ST) data.
<br/>
<div align=center>
<img src="./images/logo.png" width="300px">
</div>
<br/>


## Outline of DDATD
<br/>
<div align=center>
<img src="./images/DDATD.png" width="600px">
</div>
<br/>
The accurate detection of anomalous anatomic regions, followed by their dissection into biologically heterogeneous subdomains across multiple tissue slices, is of paramount importance in clinical diagnostics, targeted therapies and biomedical research. This procedure, which we refer to as <b>D</b>etection and <b>D</b>issection of <b>A</b>nomalous <b>T</b>issue <b>D</b>omains (DDATD), serves as the first and foremost step in a comprehensive analysis of tissues harvested from affected individuals for revealing population-level and individual-specific factors (e.g., pathogenic cell types) associated with disease developments.


## Framework of STANDS
<br/>
<div align=center>
<img src="./images/STANDS.png" width="600px">
</div>
<br/>


## Dependencies
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


## Installation
You can download the package from GitHub and install it locally:

```commandline
git clone https://github.com/Catchxu/STANDS.git
cd STANDS/
python3 setup.py install --user
```


## Source codes
All the source codes of STANDS are available on [STANDS](https://github.com/Catchxu/STANDS).


## Contributors
- [Kaichen Xu](https://github.com/Catchxu): lead developer, wrote most of the code and designed this website.
- [Kainan Liu](https://github.com/LucaFederer): developer, diverse contributions.
- Xiaobo Sun & lab: enabling guidance, support and environment.


## Tested environment
- CPU: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
- CPU Memory: 256 GB
- GPU: NVIDIA GeForce RTX 3090 
- GPU Memory: 24 GB
- System: Ubuntu 20.04.5 LTS
- Python: 3.9.15


## Citation
Coming soon.