# Detecting and dissecting anomalous anatomic regions in spatial transcriptomics with STANDS
We introduce <b>S</b>patial <b>T</b>ranscriptomics <b>AN</b>omaly <b>D</b>etection and <b>S</b>ubtyping (<b>STANDS</b>), an innovative computational method capable of integrating multimodal information, e.g., spatial gene expression, histology image and single cell gene expression, to not only delineate anomalous tissue regions but also reveal their compositional heterogeneities across multi-sample spatial transcriptomics (ST) data.
<br/>
<div align=center>
<img src="/docs/images/logo.png" width="300px">
</div>
<br/>




## Outline of DDATD
The accurate detection of anomalous anatomic regions, followed by their dissection into biologically heterogeneous subdomains across multiple tissue slices, is of paramount importance in clinical diagnostics, targeted therapies and biomedical research. This procedure, which we refer to as <b>D</b>etection and <b>D</b>issection of <b>A</b>nomalous <b>T</b>issue <b>D</b>omains (<b>DDATD</b>), serves as the first and foremost step in a comprehensive analysis of tissues harvested from affected individuals for revealing population-level and individual-specific factors (e.g., pathogenic cell types) associated with disease developments.
<br/>
<div align=center>
<img src="/docs/images/DDATD.png" width="70%">
</div>
<br/>




## Framework of STANDS
STANDS is an innovative framework built on a suite of specialized Generative Adversarial Networks (GANs) for seamlessly integrating the three tasks of DDATD. The framework consists of three components. 

<i>Component I</i> (C1) trains a GAN model on the reference dataset, learning to reconstruct normal spots from their multimodal representations of both spatial transcriptomics data and associated histology image. Subsequently, the model is applied on the target datasets to identify anomalous spots as those with unexpectedly large reconstruction deviances, namely anomaly scores.

<i>Component II</i> (C2) aims at diminishing the non-biological variations (e.g. batch effects) among anomalies via aligning target datasets in a common space. It employs two cooperative GAN models to identify pairs of reference and target spots that share similar biological contents, based on which the target datasets are aligned to the reference data space via “style-transfer”.

<i>Component III</i> (C3) fuses the embeddings and reconstruction residuals of aligned anomalous spots to serve as inputs to an iterative clustering algorithm which groups anomalies into distinct subtypes. 
<br/>
<div align=center>
<img src="/docs/images/STANDS.png" width="70%">
</div>
<br/>




## Dependencies
- anndata>=0.10.7
- dgl>=2.1.0
- networkx>=3.2.1
- numpy>=1.22.4
- pandas>=1.5.1
- Pillow>=9.4.0
- PuLP>=2.7.0
- pyemd>=1.0.0
- rpy2>=3.5.13
- scanpy>=1.10.1
- scikit_learn>=1.2.0
- scipy>=1.13.1
- torch>=2.0.0
- torchvision>=0.15.1
- tqdm>=4.64.1




## Installation
STANDS is developed as a Python package. You will need to install Python, and the recommended version is Python 3.9.

You can download the package from GitHub and install it locally:

```commandline
git clone https://github.com/Catchxu/STANDS.git
cd STANDS/
python3 setup.py install
```




## Getting Started
STANDS offers a variety of functionalities, including but not limited to: 
- Detect anomaly spots on spatial transcriptomics datasets ([tutorial](https://catchxu.github.io/STANDS/tutorial/detection/))
- Correct multi-sample batch effects from vertical or horizontal slices ([tutorial](https://catchxu.github.io/STANDS/tutorial/alignment/))
- Detect anomaly subtypes on spatial transcriptomics datasets ([tutorial](https://catchxu.github.io/STANDS/tutorial/subtype/))

Before starting the tutorial, we need to make some preparations, including: installing STANDS and its required Python packages, downloading the datasets required for the tutorial, and so on. The preparations is available at [STANDS Preparations](https://catchxu.github.io/STANDS/start/). Additionally, when dealing with multimodal data structures involving both images and gene expression matrices, we strongly recommend using a GPU and pretraining STANDS on large-scale public spatial transcriptomics datasets. This ensures faster execution of STANDS and improved performance in modules related to image feature extraction and feature fusion.

Finally, more useful and helpful information can be found at the [online documentation](https://Catchxu.github.io/STANDS/) and [tutorials](https://catchxu.github.io/STANDS/tutorial/overview/) for a quick run.




## Tested environment
### Environment 1
- CPU: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
- Memory: 256 GB
- System: Ubuntu 20.04.5 LTS
- Python: 3.9.15

### Environment 2
- CPU: Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz
- Memory: 256 GB
- System: Ubuntu 22.04.3 LTS
- Python: 3.9.18




## Citation
Coming soon.
