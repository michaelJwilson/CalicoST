# CalicoST

<p align="center">
<img src="https://github.com/raphael-group/CalicoST/blob/main/docs/_static/img/overview4_combine.png?raw=true" width="100%" height="auto"/>
</p>

CalicoST is a probabilistic model that infers allele-specific copy number aberrations and tumor phylogeography from spatially resolved transcriptomics.CalicoST has the following key features:
1. Identifies allele-specific integer copy numbers for each transcribed region, revealing events such as copy neutral loss of heterozygosity (CNLOH) and mirrored subclonal CNAs that are invisible to total copy number analysis.
2. Assigns each spot a clone label indicating whether the spot is primarily normal cells or a cancer clone with aberration copy number profile.
3. Infers a phylogeny relating the identified cancer clones as well as a phylogeography that combines genetic evolution and spatial dissemination of clones.
4. Handles normal cell admixture in SRT technologies hat are not single-cell resolution (e.g. 10x Genomics Visium) to infer more accurate allele-specific copy numbers and cancer clones.
5.  Simultaneously analyzes multiple regional or aligned SRT slices from the same tumor.

# System requirements
The package has tested on the following Linux operating systems: SpringdaleOpenEnterprise 9.2 (Parma) and CentOS Linux 7 (Core).

# Installation
First, setup a conda environment from the `environment.yml` file:
```
cd CalicoST

conda config --add channels conda-forge
conda config --add channels bioconda

conda env create -f environment.yml --name calicost_env

conda activate calicost_env
```
Next download [Eagle2](https://alkesgroup.broadinstitute.org/Eagle/) by
```
wget https://storage.googleapis.com/broad-alkesgroup-public/Eagle/downloads/Eagle_v2.4.1.tar.gz
tar -xzf Eagle_v2.4.1.tar.gz
```
Next, we need to install [Startle](https://github.com/raphael-group/startle).  Its dependencies
include [LEMON](https://lemon.cs.elte.hu/trac/lemon/wiki/InstallLinux), [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer),
perl and python3.  We do so with conda
```
conda install -c schmidt73 startle
```
or by building from source,
```
git clone --recurse-submodules https://github.com/raphael-group/startle.git

cd startle

mkdir build; cd build

cmake -DLIBLEMON_ROOT=<lemon path>\
        -DCPLEX_INC_DIR=<cplex include path>\
        -DCPLEX_LIB_DIR=<cplex lib path>\
        -DCONCERT_INC_DIR=<concert include path>\
        -DCONCERT_LIB_DIR=<concert lib path>\
        ..
make
```
Note this will install a copy of cellsnp-lite to the environment directory, which must be updated
in the config.yaml, i.e. with the output of which cellsnp-lite. 

Finally, install CalicoST using pip in the root directory with
```
pip install -e .
```
Setting up the conda environments takes around 10 minutes on an HPC head node.  Make sure to use the
[mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) backend for conda to ensure the fastest builds.

# Getting started
CalicoST requires the coordinate information of genes and SNPs, the information files for GRCh38 genome are available from either of the [example data tarball](https://github.com/raphael-group/CalicoST/tree/main/examples). Specify the information file paths, your input SRT data paths, and running configurations in `config.yaml`, and then you can run CalicoST by
```
snakemake --cores <number threads> --configfile config.yaml --snakefile calicost.smk all
```

Check out our [readthedocs](https://calicost.readthedocs.io/en/latest/) for tutorials on the simulated data and prostate cancer data.

# Run on a simulated example data
### Download data
The simulated count matrices are available from [`examples/CalicoST_example.tar.gz`](https://github.com/raphael-group/CalicoST/blob/main/examples/CalicoST_example.tar.gz).
CalicoST requires a reference SNP panel and phasing panel, which can be downloaded directly with wget and the links below
* [SNP panel](https://downloads.sourceforge.net/project/cellsnp/SNPlist/genome1K.phase3.SNP_AF5e4.chr1toX.hg38.vcf.gz?ts=gAAAAABmDbjZ1jaoDHw8fbmTQcP1y_WA9KfnTJH3aLrm0O7S4voV89YyU55O3jJdtO163_SpSBquChmB7dIl4dZ7pB-64L-W8A%3D%3D).
* [Phasing panel](http://pklab.med.harvard.edu/teng/data/1000G_hg38.zip)

Other SNP panels are available at [cellsnp-lite webpage](https://cellsnp-lite.readthedocs.io/en/latest/main/data.html).

### Run CalicoST
Gunzip the downloaded example data and replace the following paths in the provide `example_config.yaml` with those on your machine,
* calicost_dir: the path to the cloned CalicoST repository.
* eagledir: the path to the downloaded Eagle2 directory
* region_vcf: the path to the downloaded SNP panel.
* phasing_panel: the path to the downloaded and unzipped phasing panel.

To avoid falling into local maxima in CalicoST's optimization objective, we recommend running CalicoST with multiple random initializations that are specified by the `random_state` variable in `example_config.yaml`. The provided one uses five random initializations, but may be lessened for a test of the installation.

Finally, run CalicoST with
```
cd <directory of downloaded example data>

snakemake --cores 5 --configfile example_config.yaml --snakefile <calicost_dir>/calicost.smk all
```

CalicoST takes just ove an hour to complete this example when 5 cores are available.

### Understanding the results
The above snakemake run will create a folder `calicost` in the directory of downloaded example data. Within this folder, each random initialization of CalicoST generates a subdirectory of `calicost/clone*`. 

CalicoST generates the following key files of each random initialization:
* clone_labels.tsv: The inferred clone labels for each spot.
* cnv_seglevel.tsv: Allele-specific copy numbers for each clone for each genome segment.
* cnv_genelevel.tsv: The projected allele-specific copy numbers from genome segments to the covered genes.
* cnv_*_seglevel.tsv and cnv_*_genelevel.tsv: Allele-specific copy numbers when enforcing a ploidy of {diploid, triploid, tetraploid} for each genome segment or each gene.

See the following examples of the key files.
```
head -10 calicost/clone3_rectangle0_w1.0/clone_labels.tsv
BARCODES        clone_label
spot_0  2
spot_1  2
spot_2  2
spot_3  2
spot_4  2
spot_5  2
spot_6  2
spot_7  2
spot_8  0
```

```
head -10 calicost/clone3_rectangle0_w1.0/cnv_seglevel.tsv
CHR     START   END     clone0 A        clone0 B        clone1 A        clone1 B        clone2 A        clone2 B
1       1001138 1616548 1       1       1       1       1       1
1       1635227 2384877 1       1       1       1       1       1
1       2391775 6101016 1       1       1       1       1       1
1       6185020 6653223 1       1       1       1       1       1
1       6785454 7780639 1       1       1       1       1       1
1       7784320 8020748 1       1       1       1       1       1
1       8026738 9271273 1       1       1       1       1       1
1       9292894 10375267        1       1       1       1       1       1
1       10398592        11922488        1       1       1       1       1       1
```

```
head -10 calicost/clone3_rectangle0_w1.0/cnv_genelevel.tsv
gene    clone0 A        clone0 B        clone1 A        clone1 B        clone2 A        clone2 B
A1BG    1       1       1       1       1       1
A1CF    1       1       1       1       1       1
A2M     1       1       1       1       1       1
A2ML1-AS1       1       1       1       1       1       1
AACS    1       1       1       1       1       1
AADAC   1       1       1       1       1       1
AADACL2-AS1     1       1       1       1       1       1
AAK1    1       1       1       1       1       1
AAMP    1       1       1       1       1       1
```

CalicoST graphs the following plots to visualize the spatial distribution of the inferred cancer clones and allele-specific copy number profiles for each random initialization.
* ```plots/clone_spatial.pdf```: The spatial distribution of inferred cancer clones and normal regions (grey color, clone 0 by default)
* ```plots/rdr_baf_defaultcolor.pdf```: The Read Depth Ratio (RDR) and B-Allele Frequency (BAF) along the genome for each clone.  Higher RDR indicates higher total copy numbers and BAF deviations from 0.5 indicates allele imbalance due to allele-specific CNAs.
* ```plots/acn_genome.pdf```: The default allele-specific copy numbers along the genome.
* ```plots/acn_genome_*.pdf```: Allele-specific copy numbers when enforcing a ploidy of {diploid, triploid, tetraploid}.

The allele-specific copy number plots have the following color legend.
<p align="left">
<img src="https://github.com/raphael-group/CalicoST/blob/main/docs/_static/img/acn_color_palette.png?raw=true" width="20%" height="auto"/>
</p>


# Software dependencies
CalicoST uses the following command-line packages and python for extracting the BAF information
* samtools
* cellsnp-lite
* Eagle2
* pysam
* snakemake

As indicated by the provided environment.yaml and setup.py, CalicoST uses the following packages for the remaining steps to infer allele-specific copy numbers and cancer clones:
* numpy
* scipy
* pandas
* scikit-learn
* scanpy
* anndata
* numba
* tqdm
* statsmodels
* networkx
* matplotlib
* seaborn
* snakemake
