# Triton <img src="misc/logo_v1.png" width="140" align="left">
A cell free DNA (cfDNA) processing pipeline, Triton conducts fragmentomic and nucleosome-phasing coverage analyses on individual or
composite genomic regions and outputs both region-level biomarkers and nt-resolution signal profiles.
<br/><br/>


## Description
Triton conducts nucleotide-resolution profile analysis for cfDNA samples in BAM format, given a list of individual regions of interest (BED containing,
for example, promoter regions or gene bodies) or list of composite regions of interest sharing a common center (list of BED files each containing, for
example, binding locations for a single transcription factor). All fragments in each region/composite region are used to find the fragment size
distribution, coverage, and probabability of a nucleosome center at each point. GC bias correction files from Griffin† may also be incorporated
for GC correction. Fast Fourier Transforms are then used to isolate well-phased nucleosome derived signal, from which specific features are drawn.

### Outputs

Triton profiles are output as a NumPy compressed files (.npz), one for each sample, containing one object for each queried (composite) site.
Nucleotide-resolution profiles include:

    1: Depth (GC-corrected, if provided)  
    2: Nucleosome-level phased profile  
    3: Nucleosome center profile (GC-corrected, if provided)  
    4: Mean fragment size  
    5: Fragment size Shannon entropy  
    6: Fragment heterogeneity (unique fragment lengths / total fragments)  
    7: Fragment MAD (Mean Absolute Deviation)  
    8: Short:long ratio (x <= 120 / 140 <= x <= 250)  
    9: Peak locations (-1: trough, 1: peak, -2: minus-one peak, 2: plus-one peak, 3: inflection point)***  
    10: A (Adenine) frequency**  
    11: C (Cytosine) frequency**  
    12: G (Guanine) frequency**  
    13: T (Tyrosine) frequency**  
  
Triton region-level features are output as a .tsv file and include:

    site: annotation name if stacked, "name" from BED file for each region otherwise  
        ##Region-level features (fragmentation)##  
    fragment-mean: fragment lengths' mean  
    fragment-stdev: fragment lengths' standard deviation  
    fragment-mad: fragment lengths' MAD (Mean Absolute Deviation)  
    fragment-ratio: fragment lengths' short:long ratio (x <= 120 / 140 <= x <= 250)  
    fragment-entropy: fragment lengths' Shannon entropy  
        ##Region-level features (phasing)##  
    np-score: Nucleosome Phasing score  
    np-period: phased-nucleosome periodicity  
    np-amplitude: phased-nucleosome mean amplitude  
        ##Region-level features (profile-based)##  
    mean-depth: mean depth in the region (GC-corrected, if provided)  
    var-ratio: fraction of variability in the phased signal  
    plus-one-pos*: location relative to central-loc of plus-one nucleosome  
    minus-one-pos*: location relative to central-loc of minus-one nucleosome  
    plus-minus-ratio*: ratio of height of +1 nucleosome to -1 nucleosome  
    central-loc*: location of central inflection relative to window center (0)  
    central-depth*: phased signal value at the central-loc (with mean in region set to 1)  
    central-heterogeneity*: mean fragment heterogeneity value in the +/-5 bp region about the central-loc  
  
\* these features are output as np.nan if window == None  
\** sequence is based on the reference, not the reads  
\*** minus-one, plus-one, and inflection locs are only called if window != None, and supersede peak/trough

### Uses

Triton may be used either as an end point in cfDNA data analysis by outputting ready-to-use features from a given list of regions or
composite regions, or as processing step for further feature extraction from output profiles. Biomarkers reported directly from
Triton can be used to distinguish cancer lineages (see Publications) in tradtional machine learning approaches, specific profiles
may be plotted for qualitative analysis, or profile outputs may be utilized in signal-based analyses and learning structures, 
e.g. Convolutional Neural Networks (CNNs). 

### Publications

<[https://doi.org/10.1158/2159-8290.CD-22-0692](https://doi.org/10.1158/2159-8290.CD-22-0692)>

## Usage

Triton may be used as a local Python package, incoporated directly into scripts, or run on a remote cluster using the provided Snakemake.
See below for usage details:

### Inputs to Triton.py:
-n, --sample_name               : sample identifier (string)  
-i, --input                     : input .bam file (path)  
-b, --bias (*optional*)         : input-matched .GC_bias file (path, from Griffin†)  
-a, --annotation                : regions of interest as a BED file or text file containing a list of BED file paths  
                                  *if "composite" and/or "window" is specified, the BED must contain an additional  "position" column  
                                  which will be treated as the center for aligning composite regions and defining windows*  
-g, --reference_genome          : reference genome .fa file (path)  
-r, --results_dir               : directory for output (path)  
-q, --map_quality (*optional*)  : minimum read mapping quality to keep (int, default=20)  
-f, --size_range (*optional*)   : fragment size range in bp to keep (int tuple, default=(15, 500))  
-c, --cpus                      : number of CPUs to use for parallel processing of regions (int)  
-w, --window (*optional*)       : size of window to use in bp; required for composite (int, default=2000)  
-s, --composite (*optional*)    : whether to run in composite mode, treating each line of the annotation as a distinct list of regions  
                                  to overlap, or single mode, in which case the annotation should be a single BED file with each line  
                                  as a distinct region (bool, default=False)  
-d, frag_dict                   : dictionary of probable nucleosome center locations (displacements within fragments) for given fragment  
                                  lengths, as a Python binary .pkl file. Triton ships wth a pre-computed dictionary in nc_info/NCDict.pkl,  
                                  which is called by default. See nc_info for details.  

### Contained Scripts:
Triton.py | primary script containing the generate_profile() function; takes in inputs and produces outputs  
triton_helpers.py | contains helper functions called by Triton.py  
triton_cleanup.py | combines TritonFeatures.tsv output files produced by Triton when run on multiple samples; called by Snakemake  
triton_plotters.py | plotting utils and functions for TritonProfiles.npz files; use at your own discretion or modify as you see fit!  
nc_dist.py | a modified version of Triton.py for generating composite nucleosome-center profiles; see nc_info  
nc_plot.py | used after nc_dist.py to create the frag_dict and plot results  

### nc_info
Rather than exclude information about fragment length when producing nucleosome coverage signals, Triton attempts to
quantify the most probable nucleosome central coverage empirically when evaluating "nucleosome center profile" (signal output 3).
To this end "stable, tissue-independent" nucleosome positioning was garnered from NucMap (<https://ngdc.cncb.ac.cn/nucmap/>) by
overlapping 50 human iNPS peak datasets from a variety of tissue types and cell lines (<https://doi.org/10.1038/ncomms5909>)
against each other, keeping only regions represented in all samples. Triton (as nc_dist.py) was then run on the 186 remaining
high-confidence sites, using a cohort of healthy donor cfDNA from blood plasma. The resulting nc_info/NCDict.pkl represents a matrix
of fragment length vs displacement of fragment center from nucleosome center values, renormalized, so that the "weight" of each
fragment contributing to the nucleosome center profile is adjusted at each position.

In general, the results of this analysis dictate that short fragments (~150-210 bp) generally have centers coinciding with nucleosomes,
while longer fragments tend to bind nucleosome asymmetrically nearer to one end. A visualization of this phenomenon can be found in nc_info.

If the user would like to re-generate NCDict.pkl with their own site list or samples, please modify nc_dist.py and nc_plot.py as needed
and overwrite the default NCDict.pkl in future runs.

The BED file used, derived from NucMap, is also available: nc_info/hsNuc_iNPSPeak_bedops-intersect.bed

### TO RUN AS A SNAKEMAKE

Ensure the following files are up-to-date for your system and needs (defaulty values for Fred Hutch systems are included)

config/config.yaml: specify inputs as detailed above, and ensure the annotation and cluster_slurm paths are correct  
config/cluster_slurm.yaml: specify computational resources for your system  
config/samples.yaml: see example_samples.yaml for formatting; also output by default by Griffin GC correction  

Ensure the Python environment meets the requirements found in pythonversion.txt and requirements.txt; if you are on a Fred Hutch
server load the modules indicated at the head of Triton.snakefile

Run the following command to validate, then remove "-np" at the end to initiate:  
snakemake -s Triton.snakefile --latency-wait 60 --keep-going --cluster-config config/cluster_slurm.yaml --cluster "sbatch -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -c {cluster.ncpus} -n {cluster.ntasks} -o {cluster.output} -J {cluster.JobName}" -j 40 -np

### TO INSTALL AS A PACKAGE

In the repo's main directory, run:  
pip install .

Triton's primary function generate_profile() may now be imported directly into scripts, as well as helper functions.

## Requirements

See pythonversion.txt and requiremenets.txt for an up-to-date list of all package versions

## Contact
If you have any questions or feedback, please contact me at:  
**Email:** <rpatton@fredhutch.org>

## Acknowledgements
Triton is developed and maintained by Robert D. Patton in the Gavin Ha Lab, Fred Hutchinson Cancer Center.  
Anna-Lisa Doebley provided critical input and developed the GC-correction process used in Triton, originally found
in the Griffin (<https://github.com/GavinHaLab/Griffin>) pipeline.

† Griffin-based GC correction  
Triton optionally takes BAM-matched GC bias data produced by the Griffin workflow; the workflow with instructions for generating bias files can be
found at (<https://github.com/GavinHaLab/Griffin>) (when used in the snakemake as opposed to a stand-alone tool GC bias is required).

## Software License
Triton
Copyright (C) 2022 Fred Hutchinson Cancer Center

You should have received a copy of The Clear BSD License along with this program.
If not, see <https://spdx.org/licenses/BSD-3-Clause-Clear.html>.
