from setuptools import find_packages
from setuptools import setup

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='STANDS',
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
        'torch>=1.13.0',
        'dgl>=1.1.1',
        'torchvision>=0.14.1',
        'anndata>=0.10.3',
        'numpy>=1.19.2',
        'scanpy>=1.9.6',
        'scipy>=1.9.3',
        'sklearn>=0.0.post2',
        'pandas>=1.5.2',
        'squidpy>=1.2.2',
    ],
    packages=find_packages(exclude=('image')),
    zip_safe=False,
)