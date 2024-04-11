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

### Warnings
- convert_params defined multiple times
- compute_posterior_transition_sitewise defined multiple times

Warning: hmm_NB_BB_phaseswitch.py: 'compute_posterior_obs' comes from multiple modules: 'calicost.utils_hmm', 'calicost.hmm_NB_BB_nophasing', 'calicost.hmm_NB_BB_nophasing_v2'. Using 'calicost.hmm_NB_BB_nophasing_v2'.
Warning: hmm_NB_BB_phaseswitch.py: 'update_emission_params_bb_sitewise_uniqvalues' comes from multiple modules: 'calicost.utils_hmm', 'calicost.hmm_NB_BB_nophasing'. Using 'calicost.hmm_NB_BB_nophasing'.
Warning: hmm_NB_BB_phaseswitch.py: 'update_emission_params_bb_sitewise_uniqvalues_mix' comes from multiple modules: 'calicost.utils_hmm', 'calicost.hmm_NB_BB_nophasing'. Using 'calicost.hmm_NB_BB_nophasing'.
Warning: hmm_NB_BB_phaseswitch.py: 'compute_posterior_transition_sitewise' comes from multiple modules: 'calicost.utils_hmm', 'calicost.hmm_NB_BB_nophasing'. Using 'calicost.hmm_NB_BB_nophasing'.
Warning: hmm_NB_BB_phaseswitch.py: 'update_emission_params_nb_sitewise_uniqvalues_mix' comes from multiple modules: 'calicost.utils_hmm', 'calicost.hmm_NB_BB_nophasing'. Using 'calicost.hmm_NB_BB_nophasing'.
Warning: hmm_NB_BB_phaseswitch.py: 'initialization_by_gmm' comes from multiple modules: 'calicost.utils_hmm', 'calicost.hmm_NB_BB_nophasing'. Using 'calicost.hmm_NB_BB_nophasing'.
Warning: hmm_NB_BB_phaseswitch.py: 'update_startprob_sitewise' comes from multiple modules: 'calicost.utils_hmm', 'calicost.hmm_NB_BB_nophasing'. Using 'calicost.hmm_NB_BB_nophasing'.
Warning: hmm_NB_BB_phaseswitch.py: 'np_sum_ax_squeeze' comes from multiple modules: 'calicost.utils_hmm', 'calicost.hmm_NB_BB_nophasing', 'calicost.hmm_NB_BB_nophasing_v2'. Using 'calicost.hmm_NB_BB_nophasing_v2'.
Warning: hmm_NB_BB_phaseswitch.py: 'construct_unique_matrix' comes from multiple modules: 'calicost.utils_hmm', 'calicost.hmm_NB_BB_nophasing', 'calicost.hmm_NB_BB_nophasing_v2'. Using 'calicost.hmm_NB_BB_nophasing_v2'.
Warning: hmm_NB_BB_phaseswitch.py: 'update_emission_params_nb_sitewise_uniqvalues' comes from multiple modules: 'calicost.utils_hmm', 'calicost.hmm_NB_BB_nophasing'. Using 'calicost.hmm_NB_BB_nophasing'.
Warning: hmm_NB_BB_phaseswitch.py: 'mylogsumexp' comes from multiple modules: 'calicost.utils_hmm', 'calicost.hmm_NB_BB_nophasing', 'calicost.hmm_NB_BB_nophasing_v2'. Using 'calicost.hmm_NB_BB_nophasing_v2'.
Warning: hmm_NB_BB_phaseswitch.py: 'update_transition_sitewise' comes from multiple modules: 'calicost.utils_hmm', 'calicost.hmm_NB_BB_nophasing'. Using 'calicost.hmm_NB_BB_nophasing'.
Warning: hmm_NB_BB_phaseswitch.py: 'convert_params' comes from multiple modules: 'calicost.utils_hmm', 'calicost.utils_distribution_fitting', 'calicost.hmm_NB_BB_nophasing', 'calicost.hmm_NB_BB_nophasing_v2'. Using 'calicost.hmm_NB_BB_nophasing_v2'.

### NOTES
- How to pin conda: http://damianavila.github.io/blog/posts/how-to-pin-conda.html
- Apple silicon installs can be facilitated with Rosetta emulation of the x86 instruction set, see e.g. [here](https://taylorreiter.github.io/2022-04-05-Managing-multiple-architecture-specific-installations-of-conda-on-apple-M1/) - note, brew install iterm2 as duplication of the terminal app. is no longer supported.
-  poetry config virtualenvs.prefer-active-python false