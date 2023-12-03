# Detecting and dissecting anomalous anatomic regions in spatial transcriptomics with STANDS
we introduce **S**patial **T**ranscriptomics **AN**omaly **D**etection and **S**ubtyping (**STANDS**), an innovative computational method capable of integrating multimodal information, 
e.g., spatial gene expression, histology image and single cell gene expression, to not only delineate anomalous tissue regions but also reveal 
their compositional heterogeneities across multi-sample spatial transcriptomics (ST) data.


## Outline of DDATD
<br/>
<div align=center>
<img src="image/DDATD.png" width="70%">
</div>
<br/>

The accurate detection of anomalous anatomic regions, followed by their dissection into biologically heterogeneous subdomains across multiple tissue slices, is of paramount importance in clinical diagnostics, targeted therapies and biomedical research. This procedure, which we refer to as **D**etection and **D**issection of **A**nomalous **T**issue **D**omains (DDATD), serves as the first and foremost step in a comprehensive analysis of tissues harvested from affected individuals for revealing population-level and individual-specific factors (e.g., pathogenic cell types) associated with disease developments.


## Framework of STANDS
<br/>
<div align=center>
<img src="image/STANDS.png" width="70%">
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


## Installation
You can download the package from GitHub and install it locally:

```commandline
git clone https://github.com/Catchxu/STANDS.git
cd STANDS/
python3 setup.py install --user
```


## Tested environment
- CPU: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
- Memory: 256 GB
- System: Ubuntu 20.04.5 LTS
- Python: 3.9.15

## Citation
Coming soon.
