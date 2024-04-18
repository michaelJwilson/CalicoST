### Questions
- Why is it necessary to download the SNP panel and phasing panel for an example that provides the reduced matrices?
- wisdom of naming random realizations calicost/clone* etc.?
- in config, phasing_panel -> phasing_panel_dir
- wildcards: outputdir=calicost? NB prevents target rules.
- GRCh38_resources: what are these and were are they found for other references?
- genetic_map_GRCh38_merged.tab: tabix index?
- deprecate bin/* bash scripts.  These aren't used.
- what resolves outputdir variable?
- startle snakemake rule?
- sandbox deprecation

###  Warnings
- convert_params defined multiple times [hmm_NB_sharedstates.py, utils_distribution_fitting.py]
- compute_posterior_transition_sitewise defined multiple times [hmm_gaussian.py, utils_hmm.py]
- compute_posterior_obs [hmm_gaussian.py, utils_hmm.py]
- update_startprob_sitewise [hmm_gaussian.py, utils_hmm.py]
- np_sum_ax_squeeze [hmm_gaussian.py, utils_hmm.py]
- mylogsumexp [hmm_gaussian.py, utils_hmm.py]
- update_transition_sitewise [hmm_gaussian.py, utils_hmm.py]


### NOTES
- How to pin conda: http://damianavila.github.io/blog/posts/how-to-pin-conda.html
- Apple silicon installs can be facilitated with Rosetta emulation of the x86 instruction set, see e.g. [here](https://taylorreiter.github.io/2022-04-05-Managing-multiple-architecture-specific-installations-of-conda-on-apple-M1/) - note, brew install iterm2 as duplication of the terminal app. is no longer supported.
-  poetry config virtualenvs.prefer-active-python false