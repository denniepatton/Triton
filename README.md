# Triton <img src="misc/logo_v1.png" width="140" align="left">
A cell free DNA (cfDNA) processing pipeline, Triton conducts fragmentomic and phased-nucleosome coverage analyses on individual or
composite genomic regions and outputs both region-level biomarkers and nt-resolution signal profiles.
<br/><br/>


## Description
Triton conducts nucleotide-resolution profile analyses for cfDNA samples in BAM format, given a list of individual regions of interest (BED containing,
for example, promoter regions or gene bodies) or list of composite regions of interest sharing a common center (list of BED files each containing, for
example, binding locations for a single transcription factor). All fragments in each region/composite region are used to find the fragment size
distribution, coverage, and probabability of a nucleosome center at each point. GC bias correction files from Griffin† may also be incorporated
for GC correction. Fast Fourier Transforms are then used to isolate well-phased nucleosome derived signal, from which specific features are drawn.
Triton also accepts Bismark methylation caller output alignment files (see TritonMe) in which case nt-resolution and region-level methylation signals
are also output.

### Outputs

Triton signal profiles are output as NumPy compressed files (.npz), one for each sample, containing one NumPy array object for each queried
(individual or composite) site. E.g. if 100 composite site lists are passed with a window size of 2000bp, each output file will contain
100 named arrays, each with shape 2000x11.

Nucleotide-resolution profiles include:

    1: Coverage/Depth (GC-corrected, if provided)  
    2: Probable nucleosome center profile (fragment length re-weighted depth)  
    3: Phased-nucleosome profile (Fourier filtered probable nucleosome center profile)  
    4: Fragment lengths' short:long ratio (x <= 150 / x > 150)  
    5: Fragment lengths' diversity (unique fragment lengths / total fragments, i.e. multiset support / cardinality)  
    6: Fragment lengths' Shannon Entropy (normalized to window Shannon Entropy)  
    7: Peak locations (-1: trough, 1: peak, -2: minus-one peak, 2: plus-one peak, 3: inflection point)***  
    8: A (Adenine) frequency**  
    9: C (Cytosine) frequency**  
    10: G (Guanine) frequency**  
    11: T (Tyrosine) frequency**  
  
Triton region-level features are output as a .tsv file and include:

    site: annotation name if using composite sites, "name" from BED file for each region otherwise  
        ### Fragmentation Features (using all fragments in passed range/bounds) ###  
    fragment-mean: fragment lengths' mean  
    fragment-stdev: fragment lengths' standard deviation  
    fragment-median: fragment lengths' median  
    fragment-mad: fragment lengths' MAD (Median Absolute Deviation)  
    fragment-ratio: fragment lengths' short:long ratio (x <= 150 / x > 150)  
    fragment-diversity: fragment lengths' diversity (unique fragment lengths / total fragments, i.e. multiset support / cardinality)  
    fragment-entropy: fragment lengths' Shannon entropy  
        ### Phasing Features (FFT-based, using >= 146bp fragments and local peak calling) ###  
    np-score: Nucleosome Phasing Score (NPS)  
    np-period: phased-nucleosome period / mean inter-nucleosomal distance  
    np-amplitude: phased-nucleosome mean amplitude  
        ### Profiling Features (Filtered signal-based, using >= 146bp fragments and local peak calling) ###  
    mean-depth: mean depth in the region (GC-corrected, if provided)  
    var-ratio: ratio of variation in total phased signal (max signal range : max signal height)  
    plus-one-pos*: location relative to central-loc of plus-one nucleosome  
    minus-one-pos*: location relative to central-loc of minus-one nucleosome  
    plus-minus-ratio*: ratio of height of +1 nucleosome to -1 nucleosome  
    central-loc*: location of central inflection relative to window center (0)  
    central-depth*: phased signal value at the central-loc (with mean in region set to 1)  
    central-diversity*: mean fragment diversity value in the +/-5 bp region about the central-loc (with mean in region set to 1)  
    
 When run in composite mode Triton will also output a SkippedSites.bed for each samples, containing individual site
 coordinates for sites skipped due to insufficient or outlier coverage (MAD > 10 in any region). This file will share the format
 of whatever input is provided with additional "reject_reason" and "site_ID" columns. These sites may then be run in individual mode
 (in which case modifications in the "name" column may be required if identical between sites) to examine reason for removal.
  
\* these features are only output if a window is set, otherwise np.nan is reported
\** sequence is based on the reference, not the reads; in composite mode the frequency at each location is reported
\*** minus-one, plus-one, and inflection locs are only called if a window is set, and supersede peak/trough

### Uses

Triton may be used either as an end point in cfDNA data analysis by outputting ready-to-use features from a given list of regions or
composite regions, or as processing step for further feature extraction from output profiles. Biomarkers reported directly from
Triton can be used to distinguish cancer lineages (see Publications) in traditional machine learning approaches, specific profiles
may be plotted for qualitative analysis, or profile outputs may be utilized in signal-based analyses and learning structures, 
e.g. Convolutional Neural Networks (CNNs). 

### Publications

<[https://doi.org/10.1158/2159-8290.CD-22-0692](https://doi.org/10.1158/2159-8290.CD-22-0692)>

## Usage

Triton may be used as a local Python package, incorporated directly into scripts, or run on a remote cluster using the provided Snakemake(s).
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

### Inputs (extra details)

input: input .bam files are assumed to be pre-indexed with matching .bam.bai files in the same directory
bias: GC bias correction is optional but highly recommended; sample-matched .GC_bias files can be generated using Griffin's GC correction method
available at (<https://github.com/GavinHaLab/Griffin>) 
annotation: for individual mode this should be a single bed-like file (<https://www.genome.ucsc.edu/FAQ/FAQformat.html#format1>) which contains,
at minimum, the following columns: [chrom chromStart chromEnd name/Gene]. If strand is provided that will be used to orient all sites in the positive
direction; otherwise regions (including in composite sites) will be treated as belonging to the + strand. If a window is specified, a "position"
column must also be included, which defines the central point for the window. When run in composite mode, instead of passing a single bed-like
file a text file containing a list of bed-like file locations is needed; each individual file is treated as one composite-site, with reads
"piled up" across all regions based on stacking fragments in each window. Because a defined window is required for composite mode, each bed-like
file should contain the additional "position" column.
reference_genome: reference genome .fa file should match whichever build the samples were aligned to
window and composite: in individual mode (default, composite=False) window may be set or unset; in the latter case the full region from
chromStart:chromStop is used to derive signals and features, but no window-based metrics are output. In composite mode window is required.

### Contained Scripts:
Triton.py | primary script containing the generate_profile() function; takes in inputs and produces outputs  
TritonMe.py | methylation caller version of Triton for use with Bismark outputs: see TritonMe below  
triton_helpers.py | contains helper functions called by Triton.py  
triton_cleanup.py | combines TritonFeatures.tsv output files produced by Triton when run on multiple samples; called by Snakemake(s)  
triton_plotters.py | plotting utils and functions for TritonProfiles.npz files; use at your own discretion or modify as you see fit!  
triton_extractors.py | extraction utils for producing additional custom features from signal profiles; modify as you see fit!
nc_dist.py | a modified version of Triton.py for generating composite nucleosome-center profiles; see nc_info  
nc_analyze.py | used after nc_dist.py to create the frag_dict and plot results; see nc_info  

#### triton_plotters.py
triton_plotters.py is provided to allow for immediate plotting of TritonProfiles.npz outputs. It features four main plotting modes:

"all" plots all output signals (excluding nucleotide frequencies):

"signal" plots only the phased-nucleosome signal:

"RSD" plots Raw (GC-corrected) coverage, phased-nucleosome Signal, and fragment Diversity:

"TME" plots the same signals as "RSD" and also methylation signals; see TritonMe below:

triton_plotters.py also features options for grouping samples together, defining color palettes, signal normalization methods,
and restricting sites. It's a good place to start and modification is encouraged! Run Python triton_plotters.py -h for specific options and input
formatting guidance.

#### triton_extractors.py
triton_extractors.py is a bare-bones script designed to help users run their own analysis or feature extraction on TritonProfiles.npz signal outputs.
Please modify as you see fit!

### nc_info
Rather than exclude information about fragment length when producing nucleosome coverage signals, Triton attempts to
quantify the most probable nucleosome central coverage empirically when evaluating "probable nucleosome center profile" (signal output 2).
To this end "stable, tissue-independent" nucleosome positioning was garnered from NucMap (<https://ngdc.cncb.ac.cn/nucmap/>) by
overlapping 50 human iNPS peak datasets from a variety of tissue types and cell lines (<https://doi.org/10.1038/ncomms5909>)
against each other, keeping only regions represented in all samples. Triton (as nc_dist.py) was then run on the 186 remaining
high-confidence sites, using a cohort of healthy donor cfDNA from blood plasma(<https://doi.org/10.1038/s41467-019-12714-4>). The resulting
nc_info/NCDict.pkl represents a matrix of fragment length vs displacement of fragment center from nucleosome center values, renormalized,
so that the "weight" of each fragment contributing to the nucleosome center profile is adjusted at each position. The contained NCDict.pkl is
based on fitting raw counts to a triple-Gaussian: a centered distribution for capturing overlapping single nucleosomes and a symmetric, displaced
double-Gaussian for capturing dinucleosomes. Raw and fit weight-matrix visualizations, as well as raw and fit signals for specific fragment lengths,
can be found in nc_info along with the iNPS site list and information regarding the healthy donor samples used.

In general, the results of this analysis dictate that short fragments (~150-210 bp) generally have centers coinciding with nucleosomes,
while longer fragments tend to bind nucleosome asymmetrically nearer to one end or in a pattern indicative of dinucleosomal binding.

If the user would like to re-generate NCDict.pkl with their own site list or samples, please modify nc_dist.py and nc_plot.py as needed
and overwrite the default NCDict.pkl in future runs.

The BED file used, derived from NucMap, is also available: nc_info/hsNuc_iNPSPeak_bedops-intersect.bed

### Methodology

### TO RUN AS A SNAKEMAKE

Ensure the following files are up-to-date for your system and needs (default values for Fred Hutch systems are included)

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

### TritonMe (methylation reporting with Triton)

Included is an alternative Triton script, TritonMe, designed to handle Bismark alignment files which also contain methylation call information.
TritonMe runs an additional step where all reads' methylation calls are recorded at each region's location, including in composite mode, and
reported as frequency signals and region methylation levels for the four methylation contexts reported by Bismark. To use TritonMe, simply replace
Triton.py with TritonMe.py in your pipeline or utilize the TritonMe.snakefile. TritonMe outputs are:

Nucleotide-resolution profiles include:

    1: Coverage/Depth (GC-corrected, if provided)  
    2: Probable nucleosome center profile (fragment length re-weighted depth)  
    3: Phased-nucleosome profile (Fourier filtered probable nucleosome center profile)  
    4: Fragment lengths' short:long ratio (x <= 150 / x > 150)  
    5: Fragment lengths' diversity (unique fragment lengths / total fragments, i.e. multiset support / cardinality)  
    6: Fragment lengths' Shannon Entropy (normalized to window Shannon Entropy)   
    7: Peak locations (-1: trough, 1: peak, -2: minus-one peak, 2: plus-one peak, 3: inflection point)***  
    8: CpG methylation frequency (NaN if no overlapping targets)  
    9: CHG methylation frequency (NaN if no overlapping targets)  
    10: CHH methylation frequency (NaN if no overlapping targets)  
    11: CN/CHN methylation frequency (NaN if no overlapping targets)  
    12: A (Adenine) frequency**  
    13: C (Cytosine) frequency**  
    14: G (Guanine) frequency**  
    15: T (Tyrosine) frequency**  
  
Triton region-level features are output as a .tsv file and include:

    site: annotation name if using composite sites, "name" from BED file for each region otherwise  
        ### Fragmentation Features (using all fragments in passed range/bounds) ###  
    fragment-mean: fragment lengths' mean  
    fragment-stdev: fragment lengths' standard deviation  
    fragment-median: fragment lengths' median  
    fragment-mad: fragment lengths' MAD (Median Absolute Deviation)  
    fragment-ratio: fragment lengths' short:long ratio (x <= 150 / x > 150)  
    fragment-diversity: fragment lengths' diversity (unique fragment lengths / total fragments, i.e. multiset support / cardinality) 
    fragment-entropy: fragment lengths' Shannon entropy  
        ### Phasing Features (FFT-based, using >= 146bp fragments and local peak calling) ###  
    np-score: Nucleosome Phasing Score (NPS)  
    np-period: phased-nucleosome period / mean inter-nucleosomal distance  
    np-amplitude: phased-nucleosome mean amplitude  
        ### Profiling Features (Filtered signal-based, using >= 146bp fragments and local peak calling) ###  
    mean-depth: mean depth in the region (GC-corrected, if provided)  
    var-ratio: ratio of variation in total phased signal (max signal range : max signal height)  
    plus-one-pos*: location relative to central-loc of plus-one nucleosome  
    minus-one-pos*: location relative to central-loc of minus-one nucleosome  
    plus-minus-ratio*: ratio of height of +1 nucleosome to -1 nucleosome  
    central-loc*: location of central inflection relative to window center (0)  
    central-depth*: phased signal value at the central-loc (with mean in region set to 1)  
    central-diversity*: mean fragment diversity value in the +/-5 bp region about the central-loc (with mean in region set to 1)  
        ### Methylation Features (extracted from read coverage only, all fragment sizes, no GC) ###  
    cpg-methylation: fraction of methylated CpGs in the window  
    chg-methylation: fraction of methylated CHGs in the window  
    cgg-methylation: fraction of methylated CHHs in the window  
    cng-methylation: fraction of methylated CN/CHNs in the window  
    
 When run in composite mode Triton will also output a SkippedSites.bed for each samples, containing individual site
 coordinates for sites skipped due to insufficient or outlier coverage (MAD > 10 in any region). This file will share the format
 of whatever input is provided with additional "reject_reason" and "site_ID" columns. These sites may then be run in individual mode
 (in which case modifications in the "name" column may be required if identical between sites) to examine reason for removal.
  
\* these features are only output if a window is set, otherwise np.nan is reported
\** sequence is based on the reference, not the reads; in composite mode the frequency at each location is reported
\*** minus-one, plus-one, and inflection locs are only called if a window is set, and supersede peak/trough

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
