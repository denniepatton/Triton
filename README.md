# Triton <img src="misc/logo_v2.png" width="140" align="left">

As a cell-free DNA (cfDNA) processing pipeline, Triton conducts fragmentomic and phased-nucleosome coverage analyses on individual or
composite genomic regions and outputs both region-level biomarkers and nucleotide-resolution (nt-resolution) signal profiles.

_Triton_ is named for the Greek deity who served as messenger of the deep and would blow a conch shell to calm or raise the waves. Like Triton, this tool has the power to see beyond the waves and carry messages from the deep.  
<br/>

## Table of Contents
- [Description](#description)
- [Outputs](#outputs)
- [Uses](#uses)
- [Publications](#publications)
- [Usage](#usage)
  - [Inputs to Triton.py](#inputs-to-tritonpy)
  - [Contained Scripts](#contained-scripts)
  - [nc_info](#nc_info)
  - [panel_info](#panel_info)
  - [Methodology](#methodology)
  - [To Run as a Snakemake](#to-run-as-a-snakemake)
- [Example Profiles](#example-profiles)
- [Tutorial](#tutorial)
- [Requirements and Installation](#requirements-and-installation)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Description
Triton conducts bp-resolution profile analyses for cfDNA samples in BAM or CRAM format, given a list of individual regions of interest (BED file containing,
for example, promoter regions or gene bodies) or a list of composite regions of interest sharing a common center (list of BED files each containing, for
example, binding locations for a single transcription factor). All fragments in each region/composite region are used to find the fragment size
distribution, coverage, and probability of a nucleosome center at each point. GC bias correction files from Griffin† are used for GC correction by default,
though alternative methods are supported. Finally, Fast Fourier Transforms are used to isolate well-phased nucleosome-originating signal, from which nucleosome positioning features are drawn.

Updates in Version 3 (v0.3.1):
- Updated `NCDict.pkl` used for probable nucleosome center re-weighting of fragment coverage [(see nc_info)](#nc_info)
- In testing: background panel creation/subtraction modes to account for tumor purity [(see panel_info)](#panel_info)
- Methylation mode (0.2.2) has been discontinued
- Triton now outputs a fragment-end coverage profile instead of the probable nucleosome center profile (which was redundant with the phased-nucleosome profile)
- All operations in Triton now use NumPy arrays exclusively, leading to an increase in efficiency
- Updated annotation files for promoters / transcription start sites (TSS), transcript bodies, and composite transcription factor binding sites based on MANE and GTRD [(see inputs)](#inputs-extra-details)
- Window size for window and composite-window modes now defaults to ±1000 bp (2000 bp total)
- Bands of coverage without enough overlapping fragments to perform fragmentomics analyses now return 0s instead of NaNs to simplify downstream signal analyses

## Outputs
Triton bp-resolution "signal profiles" are output as NumPy compressed files (`.npz`), one for each sample, containing one NumPy array object for each queried
(individual or composite) site. For example, if 100 composite site lists are passed with a window size of 2000 bp, each output file will contain
100 named arrays, each with shape `2000×11`. Signal profiles are not output for large regions (e.g. full gene bodies) by default.

bp-resolution profiles include:

1. Coverage/Depth (GC-corrected)  
2. Fragment-end coverage/depth  
3. Phased-nucleosome profile (Fourier-filtered probable nucleosome center profile)  
4. Fragment lengths’ short:long ratio (≤150 bp / >150 bp)  
5. Fragment lengths’ diversity (unique fragment lengths ÷ total fragments)  
6. Fragment lengths’ Shannon Entropy (normalized to window Shannon Entropy)  
7. Called peaks/troughs in the phased-nucleosome profile  
   (−1: trough, 1: peak, −2: minus-one peak, 2: plus-one peak, 3: central inflection point)  
8. A (Adenine) frequency**  
9. C (Cytosine) frequency**  
10. G (Guanine) frequency**  
11. T (Thymine) frequency**  

Triton region-level features are output as a `.tsv` file and include for each single or composite site:

- **site**: annotation name (if using composite sites) or the `name` from the BED file for the queried region  
#### Fragmentation Features (using all fragment lengths in passed range/bounds)
- **fragment-mean**: mean fragment length  
- **fragment-stdev**: standard deviation of fragment lengths
- **fragment-median**: median fragment length
- **fragment-mad**: median absolute deviation of fragment lengths  
- **fragment-ratio**: short:long length ratio (≤150 / >150)  
- **fragment-diversity**: (unique fragment lengths ÷ total fragments)  
- **fragment-entropy**: Shannon entropy of fragment lengths  
#### Phasing Features (FFT-based, using ≥146 bp fragments and local peak calling)
- **np-score**: Nucleosome Phasing Score (NPS)  
- **np-period**: phased-nucleosome period (AKA mean inter-nucleosomal distance) 
- **np-amplitude**: phased-nucleosome mean amplitude  
#### Profiling Features (Filtered signal-based, using ≥146 bp fragments and local peak calling)
- **mean-depth**: mean depth in the region (GC-corrected, if provided)  
- **var-ratio**: ratio of variation in total phased signal (max signal range ÷ max signal height)  
- **plus-one-pos***: location relative to `central-loc` of plus-one nucleosome  
- **minus-one-pos***: location relative to `central-loc` of minus-one nucleosome  
- **plus-minus-ratio***: ratio of the +1 nucleosome height to the −1 nucleosome height  
- **central-loc***: location of the central inflection relative to window center (0)  
- **central-depth***: phased signal value at the `central-loc` (with mean in region set to 1)  
- **central-diversity***: mean fragment diversity within ±5 bp of the `central-loc` (with mean in region set to 1)

\* Feature only output in “window” or “composite-window” mode; otherwise `np.nan` is reported.  
\** Sequence is based on the reference, not the reads (nt frequency, in composite mode).

When run in composite mode, Triton will also output a `SkippedSites.bed` for each sample, containing individual site coordinates for sites skipped due to insufficient or outlier coverage (MAD > 10 in any region). This file will share the format of whatever input is provided, with additional `reject_reason` and `site_ID` columns.

## Uses
Triton may be used either as an endpoint in cfDNA data analysis by outputting ready-to-use features from a given list of regions or
composite regions, or as a processing step for further feature extraction from output profiles. Features reported directly from
Triton can be used in traditional machine learning approaches, or specific profiles can be plotted with accompanying scripts for
qualitative analysis. Output signal profiles may also be used in signal-based analyses or in deep learning frameworks (e.g., CNNs).

Triton features have been used to distinguish heterogeneous cancer lineages using [Keraon](https://github.com/denniepatton/Keraon), and output
profiles and features of TSSs in conjunction with features from matched gene bodies are utilized in [Proteus](https://github.com/denniepatton/Proteus) to predict individual genes’ expression directly from cfDNA (see [Publications](#publications)).

## Publications
[Nucleosome Patterns in Circulating Tumor DNA Reveal Transcriptional Regulation of Advanced Prostate Cancer Phenotypes](https://doi.org/10.1158/2159-8290.CD-22-0692)

## Usage

### Inputs to Triton.py:

```
-n, --sample_name               : sample identifier (string, required)
-i, --input                     : input .bam file (path, required)
-b, --bias                      : input-matched .GC_bias file (path, e.g., from Griffin†, required)
-a, --annotation                : regions of interest as a BED file OR text file containing a list of BED file paths (required)
-g, --reference_genome          : reference genome .fa file (path, required)
-r, --results_dir               : directory for output (path, required)
-m, --run_mode                  : run mode ("region", "window", or "composite-window"; string, required)
-q, --map_quality               : minimum read mapping quality to keep (int, default=20)
-f, --size_range                : fragment size range in bp to keep (int tuple, default=(15, 500))
-c, --cpus                      : number of CPUs to use for parallel processing of regions (int, optional)
-d, --frag_dict                 : dictionary of probable nucleosome center locations (.pkl file). Defaults to nc_info/NCDict.pkl
-s, --subtract_background_panel : path to an annotation-matched background panel for subtraction (optional)
-t, --tumor_fraction            : tumor fraction (required if -s is used; float, optional)
-p, --generate_panel            : run in background panel generation mode (bool, optional)
```

#### (extra details)

**input**  
Input `.bam` (or `.cram`) files must be pre-indexed with matching `.bam.bai` (`.cram.crai`) files in the same directory.

**bias**  
Sample-matched `.GC_bias` files can be generated using Griffin’s GC correction method (https://github.com/GavinHaLab/Griffin) or another tool producing the same format:
```
length   num_GC   smoothed_GC_bias
...
```
with all combinations of fragment length / GC content for a given sample and an associated bias.

**annotation**  
- In “region” mode, provide a single [BED-like file](https://www.genome.ucsc.edu/FAQ/FAQformat.html#format1) with columns `[chrom, chromStart, chromEnd, name/Gene]`.  
- In “window” mode, a `[position]` column is required to define the window’s center.  
- In “composite-window” mode, supply a **text file** containing paths to multiple BED-like files (each with a `[position]` column). Triton will perform a coverage “pileup” across the window for all sites in each BED file.

Example canonical annotation files are provided in `config/site_lists`:
- `MANE.GRCh38.v1.3_TranscriptBodies.bed` for region mode (full gene bodies)
- `MANE.GRCh38.v1.3_TSS.bed` for window mode (promoter regions)
- `GTRD_F1000.tsv` or `GTRD_F10000.tsv` for composite-window mode (pointing to directories of BEDs for TFBSs), which contain 1,000 and 10,000 sites, respectively, for each TFBS.

**reference_genome**  
The `.fa` file must match the build used to align your samples.

**run_mode**  
- “region” mode: analyze the entire region (`chromStart:chromStop`)  
- “window” mode: analyze ±1000 bp from the `position` index  
- “composite-window” mode: “pile up” reads from all sites in each passed file, aligning based on the  `position` index

### Contained Scripts

**Triton.py** – primary script containing the `generate_profile()` function  
**triton_helpers.py** – helper functions called by Triton.py  
**triton_cleanup.py** – combines `TritonFeatures.tsv` output from multiple samples  
**triton_plotters.py** – plotting utilities for `.npz` outputs  
**triton_extractors.py** – extraction utilities for pulling additional, custom features from signal profiles  
**triton_panel.py** – combines multiple TritonRawPanel.npz files into a single site:panel background collection  
**nc_dist.py** – a modified script for generating composite nucleosome-center profiles (see [nc_info](#nc_info))  
**nc_analyze.py** – used after nc_dist.py to create the frag_dict and plot results (see [nc_info](#nc_info))

#### triton_plotters.py
`triton_plotters.py` provides three plotting modes (`--mode / -m`): “RSD”, “all”, and “signal.”  
- “RSD” plots **R**aw (GC-corrected) coverage, **S**ignal (phased-nucleosome), and fragment **D**iversity.  
- “all” plots all Triton outputs except nucleotide frequencies.  
- “signal” plots only the phased-nucleosome signal.  

It also supports grouping samples with `--categories / -c` and color palettes with `--palette / -p`. Run `python triton_plotters.py -h` for details.

#### triton_extractors.py
`triton_extractors.py` is a simple example script to help extract or transform the `.npz` signal outputs from Triton. Users are encouraged to modify it as needed for custom analyses.

### nc_info
Instead of ignoring fragment length when producing nucleosome coverage signals, Triton empirically re-weights coverage according to each fragment’s most probable nucleosome center. This is derived from iNPS peak data across multiple human tissues/cell lines and healthy-donor cfDNA. The resulting matrix, stored as `nc_info/NCDict.pkl`, captures how fragment length maps to nucleosome center placement.

If you wish to regenerate `NCDict.pkl` with your own site lists or samples, see `nc_dist.py` and `nc_analyze.py` in `nc_info/`.

<img src="misc/NucFragDisplacements_FIT.png">

### panel_info
Triton supports background panel generation and subtraction to account for tumor purity. Given a sample with an estimated purity/tumor fraction (`-t` / `--tumor_fraction`), Triton will subtract `(1 - tfx) * background_profile` from each site’s coverage, fragment-end coverage, and other signals before processing. Preliminary testing has shown mixed results, with depth-saturation a leading concern in robust background subtraction. Playing with this method is only advised if cfDNA and background sequencing have similar, high (>100x) mean depth.

To generate your own panels, run Triton in panel-generation mode (`-p`) across healthy reference samples and combine the results with `triton_panel.py` before referencing the panel in a standard run.

### Methodology
1. Triton retrieves reads overlapping each region (`pysam` for random `.bam` access)  
2. Reads are discarded if not paired, not uniquely mapped, flagged as duplicate, or if their complete fragment lengths fall outside of the specified fragment length range (default 15–500 bp)  
3. For coverage arrays, only fragments ≥146 bp are used (fully-wrapped nucleosome length or greater)  
4. Fragment-level GC correction is applied for coverage  
5. If background subtraction is enabled, Triton subtracts `(1 - tfx) * background_profile` for each site  
6. Fourier transform filtering isolates the fundamental nucleosome phasing signal (≥146-bp period)  
7. Local peak calling yields phasing features such as nucleosome spacing and amplitude  
8. Fragment length distributions (region-level and per-bp) are used to generate the remaining fragmentomic features  

### To Run as a Snakemake
1. Update the following files to match your system:
   - `config/config.yaml` – specify inputs (annotation, cluster script path, etc.)
   - `config/cluster_slurm.yaml` – cluster resource configs
   - `config/samples.yaml` – sample info; see `example_samples.yaml`
2. If on a Fred Hutch server, load the Python modules indicated in the header of `Triton.snakefile`  
Otherwise, set-up a local environment following [Requirements and Installation](#requirements-and-installation)
3. Run the following snakemake command:
   ```
   snakemake -s Triton.snakefile --latency-wait 60 --keep-going \
   --cluster-config config/cluster_slurm.yaml \
   --cluster "sbatch -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -c {cluster.ncpus} -n {cluster.ntasks} -o {cluster.output} -J {cluster.JobName}" \
   -j 40 -np
   ```
   Removing `-np` to actually initiate jobs following validation.

## Example Profiles

Below are examples of output profiles, plotted using triton_plotters.py, illustrating typical outputs for different combinations of Triton and plotting settings.

1. The following figure shows three samples, which are ctDNA from a prostate adenocarcinoma PDX model (ARPC: LuCaP_136CR), ctDNA from a neuroendocrine-like prostate cancer PDX model (NEPC: LuCaP_145-1), and healthy donor cfDNA (NPH001). These samples were run through Triton in "composite-window" mode for a list of ~10,000 sites identified through ATAC-Seq as being "open" in ARPC but not NEPC. Profiles were plotted at sample-level (`--categories / -c` is not passed) using `--mode RSD`. Note the clean dip in nucleosome coverage seen in the ARPC line which is not reflected in NEPC or healthy. The Phased Nucleosomal Signal also smooths the raw coverage and better resolves nucleosome locations (i.e. in the healthy signal). The Fragment Diversity Index, like other spatial measures of fragment heterogeneity, shows an inverse pattern to coverage with a peak at 0 in the ARPC line, indicative of non-nucleosome associated fragments at that location.

<img src="misc/ATAC_AD-Exclusive_filtered_sig-Profiles_RSD.png">

2. The next figure again looks at PDX ctDNA from ARPC and NEPC lines, along with healthy cfDNA, but for multiple samples in each group. Here Triton was run in "window" mode for all gene TSS regions supplied in `config/site_lists/MANE.GRCh38.v1.3_TSS.bed`; this particular example is for the gene AR's promoter region. Profiles were plotted at categorical-level (using `--categories / -c`) showing the 95% confidence interval along with `--mode RSD`. This gene is expected to be active only in ARPC lines; note the canonical dip in the ARPC group slightly upstream of the 0-point along with an increase in the stability of the +1 nucleosome in the top two profiles, which is associated with active transcription. These samples are ~30x mean coverage, making fragmentation signals quite noisy in single regions.

<img src="misc/TSS_AR-Profiles_RSD.png">

3. Finally, let's look at the same groups as in 2 but once again using "composite-window" mode; this time Triton was run with TFBS composite sites (`config/site_lists/GTRD_F1000.tsv`) containing 1,000 high-confidence sites per transcription factor. Profiles were plotted at categorical-level (using `--categories / -c`) showing the 95% confidence interval along with `--mode all`, so that all Triton output signals (exlcuding reference nt frequencies and peak locations) are shown. This transcription factor, ASCL1, is active in NEPC lines but not ARPC or healthy; this is clear in the Depth and Nucleosomal Signal profiles which show a dip in coverage at overlapping ASCL1 binding sites, indicating nucleosome clearance and active binding. Fragmentation signals also show an increase in fragment heterogeneity at binding sites for NEPC samples, indicating potential ASCL1 or other non-nucleosomal protection. Fragment End Coverage similarly shows increased alignment at the +/-1 nucleosome locations relative to binding sites for NEPC samples, which may indicate a reduction in fragments spanning the binding domain.

<img src="misc/TFBS_ASCL1-Profiles_all.png">

## Tutorial

### Tutorial for Fred Hutch Servers
Below is an example workflow if you are on Fred Hutch infrastructure. Load necessary modules and then run the Triton snakefile:

```bash
# Clone Triton
git clone https://github.com/denniepatton/Triton.git
cd Triton

# Example module loads on FH systems
ml snakemake/5.19.2-foss-2019b-Python-3.7.4
ml Python/3.7.4-foss-2019b-fh1

# Run Snakemake with HPC specifics
snakemake -s Triton.snakefile --latency-wait 60 --keep-going \
--cluster-config config/cluster_slurm.yaml \
--cluster "sbatch -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -c {cluster.ncpus}" \
-j 40
```

If you prefer a more direct approach without Snakemake, simply call `Triton.py`:
```bash
python Triton.py \
  --sample_name SAMPLE_ID \
  --input /path/to/sample.bam \
  --bias /path/to/sample.GC_bias \
  --annotation config/site_lists/MANE.GRCh38.v1.3_TSS.bed \
  --reference_genome /path/to/genome.fa \
  --results_dir results/ \
  --run_mode window
```

### Tutorial for non–Fred Hutch users (Local or Other HPC)
1. **Install micromamba** (if you don’t already have it):  
   See instructions at [https://mamba.readthedocs.io](https://mamba.readthedocs.io). 
2. **Clone the Triton repository**:
   ```bash
   git clone https://github.com/denniepatton/Triton.git
   cd Triton
   ```
3. **Create the environment**:
   ```bash
   micromamba create -n Triton -f environment.yaml -c conda-forge -c bioconda
   micromamba activate Triton
   ```
4. **Local/Single-machine usage**:  
   You can run `Triton.py` directly from the command line:
   ```bash
   python Triton.py \
     --sample_name SAMPLE_ID \
     --input /path/to/sample.bam \
     --bias /path/to/sample.GC_bias \
     --annotation my_sites.bed \
     --reference_genome /path/to/genome.fa \
     --results_dir ./my_results \
     --run_mode region
   ```
5. **HPC usage (non–Fred Hutch)**:  
   - Configure your job submission scripts or cluster environment (e.g., SLURM, PBS).  
   - Adjust Snakemake’s cluster directives in the same manner as for Fred Hutch (see above example).  
   - Submit jobs with your cluster’s submission command, referencing your own cluster configuration.  

## Requirements and Installation
Triton requires standard Python libraries (e.g. NumPy, SciPy, pysam) and is compatible with Python 3.7+ (tested up to Python 3.10).

### Micromamba/Conda Environment
- We recommend building a virtual environment using the provided `environment.yaml` with micromamba
- Users can then recreate the environment exactly via:
  ```bash
  micromamba create -n Triton -f environment.yaml -c conda-forge -c bioconda
  micromamba activate Triton
  ```
- This approach **does not** require an Anaconda license; micromamba is a lightweight, open-source package manager compatible with conda-forge/bioconda.

## Contact
If you have any questions or feedback which cannot be addressed on GitHub, please contact me at <rpatton@fredhutch.org>

## Acknowledgments
Triton is developed and maintained by Robert D. Patton in the Gavin Ha Lab, Fred Hutchinson Cancer Center.  
Anna-Lisa Doebley provided input and developed the GC-correction process used in Triton, originally found
in the [Griffin](https://github.com/GavinHaLab/Griffin) pipeline.

† **Griffin-based GC correction**  
Triton takes BAM-matched GC bias data produced by the Griffin workflow; the workflow with instructions for generating bias files can be
found at [https://github.com/GavinHaLab/Griffin](https://github.com/GavinHaLab/Griffin).

## License
The MIT License (MIT)

Copyright (c) 2023 Fred Hutchinson Cancer Center

Permission is hereby granted, free of charge, to any government or not-for-profit entity, or to any person employed at one of the foregoing (each, an "Academic Licensee") who obtains a copy of this software and associated documentation files (the “Software”), to deal in the Software purely for non-commercial research and educational purposes, including the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or share copies of the Software, and to permit other Academic Licensees to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

No Academic Licensee shall be permitted to sell or use the Software or derivatives thereof in any service for commercial benefit. For the avoidance of doubt, any use by or transfer to a commercial entity shall be considered a commercial use and will require a separate license with Fred Hutchinson Cancer Center.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
