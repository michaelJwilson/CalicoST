### Questions
- Why is it necessary to download the SNP panel and phasing panel for an example that provides the reduced matrices?
- wisdom of naming random realizations calicost/clone* etc.?
- in config, phasing_panel -> phasing_panel_dir
- wildcards: outputdir=calicost? NB prevents target rules.
- GRCh38_resources: what are these and were are they found for other references?
- genetic_map_GRCh38_merged.tab: tabix index?
- deprecate bin/* bash scripts.  These aren't used.
- what resolves outputdir variable?


### NOTES
- Apple silicon installs can be facilitated with Rosetta emulation of the x86 instruction set, see e.g. [here](https://taylorreiter.github.io/2022-04-05-Managing-multiple-architecture-specific-installations-of-conda-on-apple-M1/) - note, brew install iterm2 as duplication of the terminal app. is no longer supported.