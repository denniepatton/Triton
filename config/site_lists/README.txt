This directory contains reference site lists for Triton run modes:

- region/window mode:
  - MANE.GRCh38.v1.3_TranscriptBodies.bed
  - MANE.GRCh38.v1.3_TSS.bed
- composite-window mode:
  - GTRD_F1000.tsv
  - GTRD_F10000.tsv
  - PC-ATAC.tsv

============================================================
MANE SITE LISTS
============================================================

MANE transcript info was taken from MANE.GRCh38.v1.3.summary.txt:
https://www.ncbi.nlm.nih.gov/refseq/MANE/

- MANE.GRCh38.v1.3_TranscriptBodies.bed
- MANE.GRCh38.v1.3_TSS.bed

============================================================
GTRD TFBS LISTS (GTRD_F1000 / GTRD_F10000)
============================================================

Source file:
https://gtrd.biouml.org/downloads/19.10/chip-seq/Homo%20sapiens_meta_clusters.interval.gz

Generation outline:
1) Split Homo_sapiens_meta_clusters.interval into TF-specific files using tfTitle.
2) Keep sites with peak.count > 2.
3) Merge touching/overlapping sites.
4) Remove sites overlapping exclusion regions.
5) Require each +/-5000 window to fully overlap high-mappability regions.
6) Keep TFs with at least 10,000 eligible sites.
7) Rank sites by peak.count then exp.count.
8) Export top 1,000 and top 10,000 site lists.

Representative commands (same logic used to build these files):

awk -F"\t" '{gsub("/", "-", $6); print > $6 ".txt"}' Homo_sapiens_meta_clusters.interval
for file in *.txt; do sort -t$'\t' -k1,1 -k2,2n "$file" > "${file/.txt/_sorted.txt}"; done
for file in *_raw.txt; do awk -F"\t" '$13 > 2' "$file" | bedtools merge -i - -c 4,5,6,7,8,9,10,11,12,13 -o mean,distinct,distinct,distinct,distinct,distinct,distinct,mean,mean,mean > "${file/raw/merged}"; done
for file in *_merged.txt; do bedtools intersect -v -a "$file" -b ../../Exclusions/all_exclusions_merged.bed > "${file/merged/filtered}"; done
for file in *_filtered.txt; do \
  awk -F"\t" '{print $1 "\t" $2 "\t" $3 "\t" $12 "\t" $13 "\t" int(($2+$3)/2)}' "$file" | \
  awk -F"\t" '{print $1 "\t" $6-5000 "\t" $6+5000 "\t" $4 "\t" $5 "\t" $6 "\t" $2 "\t" $3}' | \
  awk '$2 > 0' | \
  bedtools intersect -wa -f 1 -a - -b ../../Exclusions/k100.Umap.MultiTrackMappability_.90.bed | \
  awk -F"\t" '{print $1 "\t" $7 "\t" $8 "\t" $4 "\t" $5 "\t" $6}' | \
  sort -k1,1 -k2,2n | \
  sed '1s/^/chrom\tchromStart\tchromEnd\texpCount\tpeakCount\tposition\n/' > "${file/.txt/.bed}"; \
 done

for file in *_filtered.bed; do sort -t$'\t' -k5,5nr -k4,4nr "$file" | head -1000  | sort -t$'\t' -k1,1 -k2,2n | sed '1s/^/chrom\tchromStart\tchromEnd\texpCount\tpeakCount\tposition\n/' > "${file/filtered/top1000}"; done
for file in *_filtered.bed; do sort -t$'\t' -k5,5nr -k4,4nr "$file" | head -10000 | sort -t$'\t' -k1,1 -k2,2n | sed '1s/^/chrom\tchromStart\tchromEnd\texpCount\tpeakCount\tposition\n/' > "${file/filtered/top10000}"; done

To make portable file lists, avoid absolute paths (do not use realpath):
for file in GTRD_F10000/*.bed; do echo "./config/site_lists/$file" >> GTRD_F10000.tsv; done
for file in GTRD_F1000/*.bed;  do echo "./config/site_lists/$file" >> GTRD_F1000.tsv;  done

============================================================
PC-ATAC SITE LISTS
============================================================

Shipped in this repository (currently present):
- ./config/site_lists/PC-ATAC/LongATAC_ADexclusive_10000TFOverlap.bed
- ./config/site_lists/PC-ATAC/LongATAC_NEexclusive_10000TFOverlap.bed

These two files match the CD-22-0692 Methods description at a high level:
- start from phenotype-specific differential ATAC open chromatin regions,
- then restrict to regions overlapping known TFBSs from GTRD.
Reference:
https://pubmed.ncbi.nlm.nih.gov/36399432/

The paper reports 15,879 ARPC and 11,692 NEPC overlapped sites before any optional top-N reduction.

Derived filtered variants used in newer Keraon analyses:
- AD-Exclusive_filtered_sig.bed
- AD-Exclusive_filtered_top1000.bed
- NE-Exclusive_filtered_sig.bed
- NE-Exclusive_filtered_top1000.bed

Important differences for these derived variants:
- Input starts from the two LongATAC files above.
- Apply the same exclusion lists used for GTRD filtering.
- Do NOT apply the GTRD mappability==1 style restriction.
- Keep either all surviving sites (sig) or top 1000 (top1000).
