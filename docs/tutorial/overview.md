STANDS offers a variety of functionalities, including but not limited to: region anomaly detection on spatial transcriptome slices, multi-sample batch correction, detection of anomalous subdomains, and combinations of these functionalities. Here, we will provide a brief overview of these main features to help you quickly understand STANDS.


## Preparations before tutorials
Before starting the tutorial, we need to make some preparations, including: installing STANDS and its required Python packages, downloading the datasets required for the tutorial, and so on. The preparations is available at [STANDS Preparations](../start.md). Additionally, when dealing with multimodal data structures involving both images and gene expression matrices, we strongly recommend using a GPU and pretraining STANDS on large-scale public spatial transcriptomics datasets. This ensures faster execution of STANDS and improved performance in modules related to image feature extraction and feature fusion.


## Outline of tutorials
- [Tutorial 0: Pretrain STANDS basic extractor](./Pretrain.ipynb)
- [Tutorial 1: Identify cancerous domains in single ST dataset](./SingleAD.ipynb)
- [Tutorial 2: Identify cancerous domains across multiple ST datasets concurrently](./MultiAD.ipynb)
- [Tutorial 3: Align multiple ST datasets sharing identical domain types](./ShareBC.ipynb)
- [Tutorial 4: Detect anomalous subdomains with multimodal data](./subtype.ipynb)
