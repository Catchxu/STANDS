"""
Spatial Transcriptomics ANomaly Detection and Subtyping (STANDS) is an innovative computational method 
to detect anomalous tissue domains from multi-sample spatial transcriptomics (ST) data and reveal 
their biologically heterogeneous subdomains, which can be individual-specific or shared by all 
individuals. 

Detecting and characterizing anomalous anatomic regions from tissue samples from affected individuals 
are crucial for clinical and biomedical research. This procedure, which we refer to as Detection and 
Dissection of Anomalous Tissue Domains (DDATD), serves as the first and foremost step in the analysis 
of clinical tissues because it reveals factors, such as pathogenic or differentiated cell types, 
associated with the development of diseases or biological traits. Traditionally, DDATD has relied on 
either laborious expert visual inspection or computer vision algorithms applied to histology images. 
ST provides an unprecedent opportunity to enhance DDATD by incorporating spatial gene expression 
information. However, to the best of our knowledge, no existing methods can perform de novo DDATD from 
ST datasets.

STANDS is built on state-of-the-art generative models for de novo DDATD from multi-sample ST by 
integrating multimodal information including spatial gene expression, histology image, and single cell 
gene expression. STANDS concurrently fulfills DDATD's three sequential core tasks: detecting, aligning, 
and subtyping anomalous tissue domains across multiple samples. STANDS first integrates and harnesses 
multimodal information from spatial transcriptomics and associated histology images to pinpoint 
anomalous tissue regions across multiple target datasets. Next, STANDS aligns anomalies identified 
from target datasets in a common data space via style-transfer learning to mitigate their 
non-biological variations. Finally, STANDS dissects aligned anomalies into biologically heterogenous 
subtypes that are either common or unique to the target datasets. STANDS combines these processes 
into a unified framework that maintains the methodological coherence, which leads to its unparallel 
performances in DDATD from multi-sample ST.

Modules:
    read: Read single spatial data and preprocess if required.
    read_cross: Read spatial data from two sources and preprocess if required.
    read_multi: Read multiple spatial datasets and preprocess if required.
    pretrain: Pretrain STANDS using spatial data.
    evaluate: Calculate various metrics (including SGD).
"""

from .main import AnomalyDetect, KinPair, BatchAlign, Subtype
from .pretrain import pretrain
from ._read import read, read_cross, read_multi
from .evaluate import evaluate


__all__ = ['AnomalyDetect', 'BatchAlign', 'Subtype'
           'read', 'read_cross', 'read_multi',
           'pretrain', 'evaluate']