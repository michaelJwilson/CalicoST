import setuptools

setuptools.setup(
        name='calicost',
        version='v1.0.0',
        python_requires='>=3.8',
        packages=['calicost'],
        package_dir={'': 'src'},
        author='Cong Ma',
        author_email='congma@princeton.edu',
        description='Allele-specific CNAs and spatial cancer clone inference',
        long_description='CalicoST infers allele-specific copy number aberrations and cancer clones in spatially resolved transcriptomics data',
        url='https://github.com/raphael-group/CalicoST',
        install_requires=[
            'numpy=1.24.4', 
            'scipy=1.11.3', 
            'pandas=2.1.1',
            'scikit-learn=1.3.2',
            'scanpy=1.9.6',
            'anndata=0.10.3',
            'numba=0.60.0',
            'tqdm=4.66.1',
            'statsmodels=0.14.0',
            'networkx=3.2.1',
            'matplotlib=3.7.3',
            'seaborn=0.12.2',
            'pysam=0.22.1',
            'ete3=3.1.3',
            'ipykernel'
        ],
        include_package_data=True
)

