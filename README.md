# Triton <img src="misc/logo_v2.png" width="140" align="left">

As a cell-free DNA (cfDNA) processing pipeline, Triton conducts fragmentomic and phased-nucleosome coverage analyses on individual or
composite genomic regions and outputs both region-level biomarkers and nucleotide-resolution (nt-resolution) signal profiles,.

_Triton_ is named for the Greek deity who served as messenger of the deep and would blow a conch shell to calm or raise the waves. Like Triton, this tool has the power to see beyond the waves and carry messages from the deep.  
<br/><br/>

## Table of Contents
- [Description](#description)
- [Outputs](#outputs)
- [Uses](#uses)
- [Publications](#publications)
- [Usage](#usage)
  - [Inputs to Triton.py](#inputs-to-tritonpy)
  - [Inputs (extra details)](#inputs-extra-details)
  - [Contained Scripts](#contained-scripts)
    - [triton_plotters.py](#triton_plotterspy)
    - [triton_extractors.py](#triton_extractorspy)
  - [nc_info](#nc_info)
  - [panel_info](#panel_info)
  - [Methodology](#methodology)
  - [To Run as a Snakemake](#to-run-as-a-snakemake)
- [Tutorial](#tutorial)
  - [Tutorial for Fred Hutch Servers](#tutorial-for-fred-hutch-servers)
  - [Tutorial for Non–Fred Hutch Usage (Local or Other HPC)](#tutorial-for-nonfred-hutch-usage-local-or-other-hpc)
- [Requirements and Installation](#requirements-and-installation)
  - [Micromamba/Conda Environment](#micromambaconda-environment)
  - [requirements.txt or environment.yaml Best Practices](#requirementstxt-or-environmentyaml-best-practices)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Description
Triton conducts nucleotide-resolution profile analyses for cfDNA samples in BAM format, given a list of individual regions of interest (BED containing,
for example, promoter regions or gene bodies) or a list of composite regions of interest sharing a common center (list of BED files each containing, for
example, binding locations for a single transcription factor). All fragments in each region/composite region are used to find the fragment size
distribution, coverage, and probability of a nucleosome center at each point. GC bias correction files from Griffin† are used for GC correction by default,
though alternative methods are supported. Finally, Fast Fourier Transforms are used to isolate well-phased nucleosome-derived signal, from which specific features are drawn.

Updates in Version 3 (v0.3.1):
- Updated `NCDict.pkl` used for probable nucleosome center re-weighting of fragment coverage [(see nc_info)](#nc_info)
- New background panel creation/subtraction modes to account for tumor purity [(see panel_info)](#panel_info)
- Methylation mode (0.2.2) has been discontinued
- Triton now outputs a fragment end profile instead of the probable nucleosome center profile (redundant with the phased-nucleosome profile)
- All operations in Triton now use NumPy arrays exclusively, leading to an increase in efficiency
- Updated annotation files for transcription start sites (TSS), transcript bodies, and composite transcription factors [(see inputs)](#inputs-extra-details)
- Window size for window and composite-window modes now defaults to ±1000 bp (2000 bp total)
- Bands of coverage without enough overlapping fragments to perform fragmentomics analyses now return 0s instead of NaNs

## Outputs
Triton signal profiles are output as NumPy compressed files (`.npz`), one for each sample, containing one NumPy array object for each queried
(individual or composite) site. For example, if 100 composite site lists are passed with a window size of 2000 bp, each output file will contain
100 named arrays, each with shape `2000×11`.

Nucleotide-resolution profiles include:

1. Coverage/Depth (GC-corrected)  
2. Fragment end coverage/depth  
3. Phased-nucleosome profile (Fourier-filtered probable nucleosome center profile)  
4. Fragment lengths’ short:long ratio (≤150 bp / >150 bp)  
5. Fragment lengths’ diversity (unique fragment lengths ÷ total fragments)  
6. Fragment lengths’ Shannon Entropy (normalized to window Shannon Entropy)  
7. Peak locations (−1: trough, 1: peak, −2: minus-one peak, 2: plus-one peak, 3: inflection point)  
8. A (Adenine) frequency**  
9. C (Cytosine) frequency**  
10. G (Guanine) frequency**  
11. T (Thymine) frequency**  

Triton region-level features are output as a `.tsv` file and include:

- **site**: annotation name (if using composite sites) or the `name` from the BED file for each region  
  - ### Fragmentation Features (using all fragments in passed range/bounds) ###
    - **fragment-mean**: mean fragment length  
    - **fragment-stdev**: standard deviation of fragment length  
    - **fragment-median**: median fragment length  
    - **fragment-mad**: median absolute deviation of fragment lengths  
    - **fragment-ratio**: short:long ratio (≤150 / >150)  
    - **fragment-diversity**: (unique fragment lengths ÷ total fragments)  
    - **fragment-entropy**: Shannon entropy of fragment lengths  
  - ### Phasing Features (FFT-based, using ≥146 bp fragments and local peak calling) ###
    - **np-score**: Nucleosome Phasing Score (NPS)  
    - **np-period**: phased-nucleosome period / mean inter-nucleosomal distance  
    - **np-amplitude**: phased-nucleosome mean amplitude  
  - ### Profiling Features (Filtered signal-based, using ≥146 bp fragments and local peak calling) ###
    - **mean-depth**: mean depth in the region (GC-corrected, if provided)  
    - **var-ratio**: ratio of variation in total phased signal (max signal range : max signal height)  
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
Triton can be used in traditional machine learning approaches, or you can plot specific profiles with accompanying scripts for
qualitative analysis. You can also employ the output profiles in signal-based analyses or advanced learning frameworks (e.g., CNNs).

Triton features have been used to distinguish heterogeneous cancer lineages via [Keraon](https://github.com/denniepatton/Keraon), and output
profiles of TSSs in conjunction with features from matched gene bodies are utilized in [Proteus](https://github.com/denniepatton/Proteus) to predict individual genes’ expression directly from cfDNA (see [Publications](#publications)).

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

### Inputs (extra details)

**input**  
Input `.bam` files must be pre-indexed with matching `.bam.bai` files in the same directory.

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
- `MANE.GRCh38.v1.3_TranscriptBodies.bed` for region mode
- `MANE.GRCh38.v1.3_TSS.bed` for window mode
- `GTRD_F1000.tsv` or `GTRD_F10000.tsv` for composite-window mode (pointing to directories of BEDs)

**reference_genome**  
The `.fa` file must match the build used to align your samples.

**run_mode**  
- “region” mode: analyze the entire region (`chromStart:chromStop`)  
- “window” mode: analyze ±1000 bp from the `position` index  
- “composite-window” mode: “pile up” reads from multiple regions around each file’s `position`.

### Contained Scripts

**Triton.py** – primary script containing the `generate_profile()` function.  
**triton_helpers.py** – helper functions called by Triton.py.  
**triton_cleanup.py** – combines `TritonFeatures.tsv` output from multiple samples.  
**triton_plotters.py** – plotting utilities for `.npz` outputs.  
**triton_extractors.py** – extraction utilities for additional custom features from signal profiles.  
**triton_panel.py** – combines multiple TritonRawPanel.npz files into a single site:panel background collection.  
**nc_dist.py** – a modified script for generating composite nucleosome-center profiles (see [nc_info](#nc_info)).  
**nc_analyze.py** – used after nc_dist.py to create the frag_dict and plot results (see [nc_info](#nc_info)).

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

<img src="misc/NucFragDisplacements_FIT.png" width="450">

### panel_info
Triton supports background panel generation and subtraction to account for tumor purity. Given a sample with an estimated purity/tumor fraction (`-t` / `--tumor_fraction`), Triton will subtract `(1 - tfx) * background_profile` from each site’s coverage, fragment-end coverage, and other signals before processing. 

Precomputed panels exist for gene bodies, TSS, and TFBS-based annotations (not included in this repo due to size). To generate your own, run Triton in panel-generation mode (`-p`) across healthy reference samples and combine the results with `triton_panel.py`.

### Methodology
1. Triton retrieves reads from each region (`pysam` for random `.bam` access).  
2. Reads must be paired, uniquely mapped, not flagged as duplicates, and within the specified fragment length range (default 15–500 bp).  
3. For coverage arrays, only fragments ≥146 bp are used (approx. fully wrapped nucleosomes).  
4. Fragment-level GC correction is applied (if bias file is provided).  
5. If background subtraction is enabled, Triton subtracts `(1 - tfx) * background_profile` for each site.  
6. Fourier transform filtering isolates the fundamental nucleosome phasing signal (≥146-bp period).  
7. Local peak calling yields phasing features such as nucleosome spacing and amplitude.  
8. Fragment length distributions (region-level and per-bp) generate the remaining fragmentomic features.  

### To Run as a Snakemake
1. Update the following files to match your system:
   - `config/config.yaml` – specify inputs (annotation, cluster script path, etc.)
   - `config/cluster_slurm.yaml` – cluster resource configs
   - `config/samples.yaml` – sample info; see `example_samples.yaml`
2. If on a Fred Hutch server, load the Python modules indicated in `Triton.snakefile`.
3. Validate and then run:
   ```
   snakemake -s Triton.snakefile --latency-wait 60 --keep-going \
   --cluster-config config/cluster_slurm.yaml \
   --cluster "sbatch -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -c {cluster.ncpus} -n {cluster.ntasks} -o {cluster.output} -J {cluster.JobName}" \
   -j 40 -np
   ```
   Remove `-np` to actually initiate jobs.

## Tutorial

### Tutorial for Fred Hutch Servers
Below is an example workflow if you are on Fred Hutch infrastructure. Load necessary modules (Python or micromamba) and run Triton/Snakemake:

```bash
# Example environment load on FH systems
module load Python/3.7.4-foss-2019b-fh1
# or if using a micromamba/conda approach on FH:
module load micromamba   # (if available)

# Clone Triton
git clone https://github.com/YourUsername/Triton.git
cd Triton

# (Optional) Create environment on Fred Hutch if you have micromamba
micromamba create -n Triton -f environment.yaml -c conda-forge -c bioconda
micromamba activate Triton

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

### Tutorial for Non–Fred Hutch Usage (Local or Other HPC)
1. **Install micromamba** (if you don’t already have it):  
   See instructions at [https://mamba.readthedocs.io](https://mamba.readthedocs.io). 
2. **Clone the Triton repository**:
   ```bash
   git clone https://github.com/YourUsername/Triton.git
   cd Triton
   ```
3. **Create the environment** (using only `bioconda` and `conda-forge` channels):
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

Feel free to adapt the CPU (`--cpus`) or memory usage in your HPC scripts.

## Requirements and Installation
Triton requires standard Python libraries (NumPy, SciPy, pysam) and is compatible with Python 3.7+ (tested up to Python 3.10).

### Micromamba/Conda Environment
- We recommend distributing a `environment.yaml` (or `environment.yml`) file at the root of your GitHub repository.  
- Users can then recreate the environment exactly via:
  ```bash
  micromamba create -n Triton -f environment.yaml -c conda-forge -c bioconda
  micromamba activate Triton
  ```
- This approach **does not** require an Anaconda license; micromamba is a lightweight, open-source package manager compatible with conda-forge/bioconda.

### requirements.txt or environment.yaml Best Practices
- **`environment.yaml`**: Allows you to pin exact versions and specify channels (e.g., `bioconda`, `conda-forge`). Anyone can run `micromamba create -n Triton -f environment.yaml` to replicate the environment.
- **`requirements.txt`**: Often used for pip-based installations. You can include minimal requirements here if some users prefer pip. Note that certain packages (e.g., `pysam`) may install more smoothly via conda channels.  
- **Repository Integration**: Place `environment.yaml` and/or `requirements.txt` in your root directory. Add a short note in the README pointing to them for quick setup instructions.

## Contact
If you have any questions or feedback, please contact:
**Email:** <rpatton@fredhutch.org>

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
