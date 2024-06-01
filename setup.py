from setuptools import find_packages
from setuptools import setup

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='stands',
    version='1.0.0',
    description='Detecting and dissecting anomalous anatomic regions in spatial transcriptomics with STANDS',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Kaichen Xu',
    author_email='Kaichenxu@stu.zuel.edu.cn',
    url='https://github.com/Catchxu/STANDS',
    license='GPL v3',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.8",
    install_requires=[
        'anndata>=0.10.7',
        'dgl>=2.1.0',
        'networkx>=3.2.1',
        'numpy>=1.22.4',
        'pandas>=1.5.1',
        'pillow>=9.4.0',
        'pulp>=2.7.0',
        'pyemd>=1.0.0',
        'rpy2>=3.5.13',
        'scanpy>=1.10.1',
        'sklearn>=1.2.0',
        'scipy>=1.13.1',
        'torch>=2.0.0',
        'torchvision>=0.15.1',
        'tqdm>=4.64.1'
    ],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    zip_safe=False,
)