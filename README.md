# Triton <img src="misc/logo_v2.png" width="140" align="left">

As a cell-free DNA (cfDNA) processing pipeline, Triton conducts fragmentomic and phased-nucleosome coverage analyses on individual or
composite genomic regions and outputs both region-level biomarkers and nucleotide-resolution signal profiles.

_Triton_ is named for the Greek deity who served as messenger of the deep and would blow a conch shell to calm or raise the waves. Like Triton, this tool has the power to see beyond the waves and carry messages from the deep.  

**Current Version: v2.0.0** 
<br/>

## Table of Contents
- [Description](#description)
- [What's New in v2.0](#whats-new-in-v20)
- [Outputs](#outputs)
- [Feature Details and Interpretation](#feature-details-and-interpretation)
- [Uses](#uses)
- [Publications](#publications)
- [Usage](#usage)
  - [Inputs to Triton.py](#inputs-to-tritonpy)
  - [Contained Scripts](#contained-scripts)
  - [nc_info](#nc_info)
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

## What's New in v2.0

Version 2.0.0 represents a major release with significant improvements to accuracy, normalization, and biological interpretability:

### Core Improvements
- **Enhanced GC Correction**: GC bias correction now applies to all fragmentomic analyses, not just coverage, improving accuracy of fragment length distributions and derived metrics
- **Updated Nucleosome Center Weighting**: New empirically-derived weighting matrix (`NCDict.pkl`) better reflects nucleosome dyad positioning across fragment lengths

### Normalization & Weighting
- **Composite Mode Improvements**: 
   - Sites now weighted by sqrt(N_fragments) instead of N_fragments, better balancing reliability vs. over-dominance of high-depth sites
   - Composite-window flank normalization now uses a stable ratio-of-sums estimator (robust to low-depth per-site flank noise)
- **Flanking Region Strategy**: Expanded flanking regions (5000 bp default) used exclusively for normalization, excluding the analysis window itself for cleaner signal
- **Window-Normalized Depth**: Now reports both window-normalized depth and central depth metrics (aligned with Griffin methodology)

### Feature & Output Changes
- **Subnucleosomal Threshold**: Fragment length ratio and related metrics now use biologically-motivated 147 bp threshold (minimum nucleosome wrapping length) instead of 150 bp
- **Fragment Orientation Asymmetry**: New signal track quantifies directional cleavage patterns: (5' ends - 3' ends) / (5' ends + 3' ends)
- **Complete Fragment Distributions**: Outputs site-level GC-corrected fragment length distributions for custom downstream analyses
- **Improved Centering**: Central features now computed at position 0 (window center) rather than approximate center, for consistent alignment

### Technical Updates
- **Python 3.13 Support**: Tested and compatible with Python 3.13
- **Strand Handling**: For composite sites without strand information, uses mirrored averaging (both orientations with 1/2 weight each) instead of random assignment
- **Simplified Options**: Background panel generation/subtraction mode removed (was experimental with mixed results)
- **Robust Edge Cases**: Better handling of low-coverage regions and edge cases to prevent NaN propagation

### Performance
- **Memory Optimization**: More efficient array operations and reduced memory overhead for large-scale analyses
- **Vectorized Operations**: Expanded use of NumPy vectorization for faster processing

## Outputs

### Region-Level Features (`_TritonFeatures.tsv`)

Triton region-level features are output as a `.tsv` file and include for each single or composite site:

- **site**: annotation name (composite sites) or `name` from the BED file (individual regions)

#### Fragment Length (FL) Features
*Computed using all fragments within specified length range*

- **fl-mean**: Mean fragment length (bp)
- **fl-stdev**: Standard deviation of fragment lengths (bp)
- **fl-skew**: Skewness of fragment length distribution
- **fl-kurtosis**: Excess kurtosis of fragment length distribution
- **fl-subnucleosomal-ratio**: log₂(fraction < 147 bp / fraction ≥ 147 bp)
- **fl-entropy**: Shannon entropy (Pielou's evenness) of fragment lengths (5-bp binned)
- **fl-gini-simpson**: Gini-Simpson diversity index of fragment lengths (5-bp binned)

#### Phased-Nucleosome (PN) Features
*FFT-based metrics using fragments ≥147 bp*

- **pn-compaction-score**: log₂(amplitude 146–177 bp / amplitude 177–207 bp) — higher indicates compact chromatin
- **pn-mean-spacing**: Mean inter-nucleosomal distance (bp)
- **pn-mean-amplitude**: Mean amplitude of phased nucleosome peaks

#### Profiling Features
*Signal-based metrics computed from bp-resolution tracks*

- **mean-region-depth**: Mean GC-corrected coverage (absolute weighted average in composite mode; raw in region/window mode)
- **central-depth** *: PN signal at window center (mean ±5 bp, normalized by flanking PN signal)
- **central-entropy** *: FL entropy at window center (mean ±5 bp, robust z-score vs. flanking signal)
- **central-gini-simpson** *: FL Gini-Simpson at window center (mean ±5 bp, robust z-score vs. flanking signal)
- **window-depth**: PN signal mean across window (normalized by flanking signal)
- **window-entropy**: FL entropy mean across window (robust z-score vs. flanking signal)
- **window-gini-simpson**: FL Gini-Simpson mean across window (robust z-score vs. flanking signal)

\* Window/composite-window mode only; `np.nan` otherwise

### Signal Profiles (`_TritonProfiles.npz`)

Triton outputs bp-resolution "signal profiles" as NumPy compressed files (`.npz`), one per sample, containing one array per queried site. For example, 100 composite sites with 2000 bp windows produce 100 arrays of shape `2000×8` (composite) or `2000×12` (window/region). Profiles are not output for regions >5 kb to conserve memory.

**Signal tracks (all modes):**
1. **Depth**: GC-corrected coverage, normalized by flanking mean
2. **Fragment End Coverage**: Fragment end density (5' + 3' ends) / total fragments
3. **Fragment End Orientation Asymmetry**: (5' ends - 3' ends) / (5' + 3' ends) — directional cleavage signature
4. **PN profile**: Phased-nucleosome signal (FFT low-pass filtered, normalized by flanking mean)
5. **FL subnucleosomal ratio**: log₂(< 147 bp / ≥ 147 bp) at each position
6. **FL entropy**: Shannon entropy (robust z-score vs. flanking median and MAD)
7. **FL Gini-Simpson**: Diversity index (robust z-score vs. flanking median and MAD)
8. **PN peak locations**: Local maxima/minima (1: peak, -1: trough)

**Additional tracks (window/region mode only):**

9. **A**: Adenine (one-hot encoded, based on reference sequence)
10. **C**: Cytosine (one-hot encoded, based on reference sequence)
11. **G**: Guanine (one-hot encoded, based on reference sequence)
12. **T**: Thymine (one-hot encoded, based on reference sequence)

### Additional Outputs

- **Fragment Length Distributions** (`_FragmentLengths.npz`): Complete GC-corrected fragment length probability distributions for each site
- **Skipped Sites** (`_SkippedSites.bed`): Individual site coordinates excluded due to insufficient coverage (< 80% bases covered) or outlier metrics (MAD > 10), with rejection reason and site ID columns

## Feature Details and Interpretation

### Region-Level Features

The following table details each region-level feature with biological context:

| Feature                     | Description                                                               | Biological Interpretation                                                                                                                                                                                                                                                          |
| --------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **fl-mean**                 | Mean fragment length (bp).                                                | Typically ~167 bp in healthy cfDNA (nucleosome + linker). Shorter means increased cleavage within non-nucleosomal regions (e.g., TF-bound or open chromatin). In cancer or highly active chromatin, necrotic degradation and transcriptional activity both bias fragments shorter. |
| **fl-stdev**                | Standard deviation of fragment lengths.                                   | Reflects consistency of nucleosome wrapping. Low in well-phased, inactive regions (tight nucleosomal protection), high in transcriptionally active or disrupted chromatin.                                                                                                         |
| **fl-skew**                 | Skewness of fragment length distribution.                                 | Negative skew indicates a heavier short-fragment tail (increased open/active chromatin); positive skew indicates excess long fragments (compact chromatin or high nucleosome stability).                                                                                           |
| **fl-kurtosis**             | Excess kurtosis of fragment length distribution.                          | High (leptokurtic) = strong tails with outliers (heterogeneous fragmentation, cancer); low (platykurtic) = uniform, well-phased nucleosomal fragments (healthy).                                                                                                                   |
| **fl-subnucleosomal-ratio** | log₂ (frac < 147 bp / frac ≥ 147 bp).                                     | Enrichment of sub-nucleosomal (<147 bp) fragments marks open chromatin and non-nucleosomal protection (TF binding, nuclease hypersensitivity).                                                                                                                                     |
| **fl-entropy**              | Shannon entropy (Pielou-normalized) of fragment-length distribution.      | Quantifies uncertainty/diversity in observed fragment lengths. Higher entropy = greater heterogeneity and open chromatin; lower entropy = regular nucleosomal spacing.                                                                                                             |
| **fl-gini-simpson**         | 1 − ∑ pᵢ² diversity index of fragment lengths.                            | Robust at low depth; similar interpretation to entropy. High values = diverse fragment lengths → open chromatin or TF activity.                                                                                                                                                    |
| **pn-compaction-score**     | log₂ (amplitude 146–177 bp / amplitude 177–207 bp) from FFT on PN signal. | High = compact, inactive chromatin with strong 10.5 bp periodicity. Low = open or disorganized nucleosome arrays typical of active or cancer regions.                                                                                                                              |
| **pn-mean-spacing**         | Average inter-nucleosomal distance (bp) from PN signal.                   | Longer spacing → more open chromatin; shorter spacing → denser nucleosome packing.                                                                                                                                                                                                 |
| **pn-mean-amplitude**       | Mean amplitude of periodic PN peaks.                                      | High amplitude = well-phased nucleosomes (stable/quiet); low amplitude = dynamic or disrupted nucleosomes (active).                                                                                                                                                                |
| **mean-region-depth**       | GC-corrected mean coverage.                                               | Indicates read support. In composite mode, reflects average per-site depth across all contributing windows.                                                                                                                                                                        |
| **central-depth** *         | PN signal mean ± 5 bp around the center.                                  | Low or negative = nucleosome vacancy or accessible region.                                                                                                                                                                                                                         |
| **central-entropy** *       | FL entropy mean ± 5 bp around the center.                                 | High = increased heterogeneity, suggesting TF or remodeler binding.                                                                                                                                                                                                                |
| **central-gini-simpson** *  | FL diversity mean ± 5 bp around the center.                               | Same as entropy: elevated in accessible, dynamic regions.                                                                                                                                                                                                                          |
| **window-depth**            | PN signal mean across full window.                                        | Low or negative = nucleosome depletion.                                                                                                                                                                                                                                            |
| **window-entropy**          | FL entropy mean across window.                                            | High = heterogeneous or active region.                                                                                                                                                                                                                                             |
| **window-gini-simpson**     | FL diversity mean across window.                                          | High = variable fragment lengths → open chromatin.                                                                                                                                                                                                                                 |

\* Window/composite-window mode only

### Signal Profiles

The following table details each bp-resolution signal track:

| Signal                                        | Description                                                                                   | Biological Interpretation                                                                                    |
| --------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **depth**                                     | GC-corrected coverage normalized to flanks.                                                   | Relative local coverage; correlates with fragment abundance.                                                 |
| **EPC (fragment end coverage)**               | End-position counts of cfDNA fragments.                                                       | Peaks mark nuclease-accessible linker regions; dips indicate nucleosome cores.                               |
| **ASYM (fragment end orientation asymmetry)** | Difference between 5′ and 3′ fragment-end coverage (oriented).                                | Sign changes (peak→dip) define nucleosome footprints; large magnitude = strong directional cleavage pattern. |
| **PN profile**                                | Coverage re-weighted by per-fragment nucleosome-dyad probability and low-pass filtered (FFT). | Highlights constitutive nucleosome phasing and chromatin compaction.                                         |
| **FL subnucleosomal ratio**                   | log₂ (x < 147 / x ≥ 147) at each bp.                                                          | Local enrichment of sub-nucleosomal fragments → open chromatin, TF binding.                                  |
| **FL Shannon entropy / Pielou evenness**      | Robust z-score vs median ± MAD of flanking signal.                                            | Elevated = heterogeneous fragment lengths (dynamic chromatin).                                               |
| **FL Gini-Simpson index**                     | Robust z-score vs flanks.                                                                     | Same interpretation as entropy; robust at lower read depth.                                                  |
| **PN peak locations**                         | Detected PN local maxima.                                                                     | Mark well-phased nucleosomes and enable spacing/compaction quantification.                                   |

### Complementary Metric Categories

Triton's features capture distinct aspects of chromatin biology:

| Category                                      | Representative Metrics                                  | Distinct Insight                                                                                                           |
| --------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Absolute fragment length**                  | fl-mean, fl-subnucleosomal-ratio                        | Indicates typical protection length and cleavage origin.                                                                   |
| **Heterogeneity / variability**               | fl-stdev, fl-entropy, fl-gini-simpson, fl-kurtosis      | Capture spread and diversity, which distinguish *mixed* chromatin states even if means are equal.                          |
| **Shape / asymmetry**                         | fl-skew, fl-kurtosis                                    | Reveal bias toward short or long tails—detect asymmetric cleavage activity.                                                |
| **Chromatin phasing / compaction**            | pn-compaction-score, pn-mean-spacing, pn-mean-amplitude | Operate in frequency domain, independent of fragment length histogram; reflect higher-order nucleosome array organization. |
| **Occupancy / accessibility (local signals)** | depth, EPC, ASYM, PN profile                            | Resolve spatial context (linker vs nucleosome center) unavailable to region-level stats.                                   |

### Cohort-Level Analyses

**Important Note**: When performing comparisons across samples or patients within a cohort, it is strongly recommended to apply **site-level z-scoring** to normalize for technical variation and batch effects. This standardization ensures that biological differences, rather than sequencing depth or sample-specific biases, drive downstream analyses and model predictions.

## Uses
Triton may be used either as an endpoint in cfDNA data analysis by outputting ready-to-use features from a given list of regions or
composite regions, or as a processing step for further feature extraction from output profiles. Features reported directly from
Triton can be used in traditional machine learning approaches, or specific profiles can be plotted with accompanying scripts for
qualitative analysis. Output signal profiles may also be used in signal-based analyses or in deep learning frameworks (e.g., CNNs).

Triton features have been used to distinguish heterogeneous cancer lineages using [Keraon](https://github.com/denniepatton/Keraon), and output
profiles and features of TSSs in conjunction with features from matched gene bodies are utilized in [Proteus](https://github.com/denniepatton/Proteus) to predict individual genes' expression directly from cfDNA (see [Publications](#publications)).

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
```

#### (extra details)

**input**  
Input `.bam` (or `.cram`) files must be pre-indexed with matching `.bam.bai` (`.cram.crai`) files in the same directory.

**bias**  
Sample-matched `.GC_bias` files can be generated using Griffin's GC correction method (https://github.com/GavinHaLab/Griffin) or another tool producing the same format:
```
length   num_GC   smoothed_GC_bias
...
```
with all combinations of fragment length / GC content for a given sample and an associated bias.

**annotation**  
- In "region" mode, provide a single [BED-like file](https://www.genome.ucsc.edu/FAQ/FAQformat.html#format1) with columns `[chrom, chromStart, chromEnd, name/Gene]`.  
- In "window" mode, a `[position]` column is required to define the window's center.  
- In "composite-window" mode, supply a **text file** containing paths to multiple BED-like files (each with a `[position]` column).

Example canonical annotation files are provided in `config/site_lists`:
- `MANE.GRCh38.v1.3_TranscriptBodies.bed` for region mode (full gene bodies)
- `MANE.GRCh38.v1.3_TSS.bed` for window mode (promoter regions)
- `GTRD_F1000.tsv` or `GTRD_F10000.tsv` for composite-window mode (pointing to directories of BEDs for TFBSs), which contain 1,000 and 10,000 sites, respectively, for each TFBS.

**reference_genome**  
The `.fa` file must match the build used to align your samples.

**run_mode**  
- "region" mode: analyze the entire region (`chromStart:chromStop`)  
- "window" mode: analyze ±1000 bp from the `position` index  
- "composite-window" mode: aggregate reads from all sites in each passed file, aligning based on the  `position` index

### Contained Scripts

**Triton.py** – primary script containing the `generate_profile()` function  
**triton_helpers.py** – helper functions called by Triton.py  
**triton_cleanup.py** – combines `TritonFeatures.tsv` output from multiple samples  
**triton_plotters.py** – plotting utilities for `.npz` outputs  
**triton_extractors.py** – extraction utilities for pulling additional, custom features from signal profiles  
**nc_dist.py** – a modified script for generating composite nucleosome-center profiles (see [nc_info](#nc_info))  
**nc_analyze.py** – used after nc_dist.py to create the frag_dict and plot results (see [nc_info](#nc_info))

#### triton_plotters.py
`triton_plotters.py` provides three plotting modes (`--mode / -m`): "RSD", "all", and "signal."  
- "RSD" plots **R**aw (GC-corrected) coverage, **S**ignal (phased-nucleosome), and fragment **D**iversity.  
- "all" plots all Triton outputs except nucleotide frequencies.  
- "signal" plots only the phased-nucleosome signal.  

It also supports grouping samples with `--categories / -c` and color palettes with `--palette / -p`. Run `python triton_plotters.py -h` for details.

#### triton_extractors.py
`triton_extractors.py` is a simple example script to help extract or transform the `.npz` signal outputs from Triton. Users are encouraged to modify it as needed for custom analyses.

### nc_info
Instead of ignoring fragment length when producing nucleosome coverage signals, Triton empirically re-weights coverage according to each fragment's most probable nucleosome center. This is derived from iNPS peak data across multiple human tissues/cell lines and healthy-donor cfDNA. The resulting matrix, stored as `nc_info/NCDict.pkl`, captures how fragment length maps to nucleosome center placement.

If you wish to regenerate `NCDict.pkl` with your own site lists or samples, see `nc_dist.py` and `nc_analyze.py` in `nc_info/`.

<img src="misc/NucFragDisplacements_FIT.png">

### Methodology
1. Triton retrieves reads overlapping each region (`pysam` for random `.bam` access)  
2. Reads are discarded if not paired, not uniquely mapped, flagged as duplicate, or if their complete fragment lengths fall outside of the specified fragment length range (default 15–500 bp)  
3. For coverage arrays, only fragments ≥147 bp are used (minimum nucleosome wrapping length)  
4. Fragment-level GC correction is applied to all fragments for both coverage and fragmentomics analyses
5. Nucleosome center re-weighting applied based on fragment length using empirical probability matrix
6. Flank normalization: composite-window uses a ratio-of-sums estimator across sites; window/region use post-aggregation normalization
7. Fourier transform filtering isolates the fundamental nucleosome phasing signal (periods ≥147 bp)  
8. Local peak calling yields phasing features such as nucleosome spacing and amplitude  
9. Fragment length distributions (region-level and per-bp) are used to generate the remaining fragmentomic features  

#### Composite Mode Normalization Details
In composite-window mode, signals undergo careful normalization to ensure biological comparability:

1. **Flank normalization (stable)**: Depth and nucleosome-center (nc_signal) profiles are normalized to flanks using a ratio-of-sums estimator
   across sites: (Σ wᵢ · signalᵢ) / (Σ wᵢ · flankMeanᵢ). This avoids instability when per-site flank means are noisy at low depth.
2. **Fragment-based probability normalization**: Fragment length distributions, fragment length profiles, and fragment end profiles converted to probability distributions (normalized by totals) so contribution is proportional to fragmentation patterns rather than absolute counts
3. **Weighting**: Each site receives weight w = sqrt(N_frag_raw_ROI) (raw fragments fully contained in the ROI = window + flanks) to balance reliability vs. over-dominance
4. **Weighted aggregation**: Numerators and denominators are accumulated separately for depth/nc_signal; fragment-based probability distributions are accumulated as weighted sums
5. **Final normalization**: Depth/nc_signal are finalized via the ratio-of-sums; fragment-based probability distributions are divided by total weight W = Σw

This ensures sites contribute based on fragmentation patterns rather than raw counts, with weighting proportional to sqrt(total_fragments) to prevent high-depth sites from dominating the composite signal.

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
1. **Install micromamba** (if you don't already have it):  
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
   - Adjust Snakemake's cluster directives in the same manner as for Fred Hutch (see above example).  
   - Submit jobs with your cluster's submission command, referencing your own cluster configuration.  

## Requirements and Installation
Triton requires standard Python libraries (e.g. NumPy, SciPy, pysam) and is compatible with Python 3.7+ (tested up to Python 3.13).

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

Copyright (c) 2025 Fred Hutchinson Cancer Center

Permission is hereby granted, free of charge, to any government or not-for-profit entity, or to any person employed at one of the foregoing (each, an "Academic Licensee") who obtains a copy of this software and associated documentation files (the "Software"), to deal in the Software purely for non-commercial research and educational purposes, including the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or share copies of the Software, and to permit other Academic Licensees to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

No Academic Licensee shall be permitted to sell or use the Software or derivatives thereof in any service for commercial benefit. For the avoidance of doubt, any use by or transfer to a commercial entity shall be considered a commercial use and will require a separate license with Fred Hutchinson Cancer Center.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
