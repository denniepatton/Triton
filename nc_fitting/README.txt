### Guide to reproducing the config/NucMapp_iNPS_1000.bed file

### Download all (available as of 09/22/2025) iNPS peak data from NucMap
(see iNPS_peaks/Homo_sapiens.nucleosome.iNPSPeak.bed_FileHistory.txt for record of used samples)

wget -r -np -nH --cut-dirs=6 \
  -e robots=off -R 'index.html*' \
  --accept-regex '.*\.bed(\.gz)?$' \
  -P iNPS_peaks \
  https://download.cncb.ac.cn/nucmap/organisms/latest/Homo_sapiens/data_type/iNPS_peaks/

### Extract **MainPeak** only and make **1-bp summits**
Keep lines whose 4th column ends with `:MainPeak` (not `MainPeak+Shoulder`), compute the integer midpoint, and tag each row with a **sample id** from the filename.

for f in iNPS_peaks/*.bed.gz; do
  base=$(basename "$f" .bed.gz)
  sample="$base"
  zcat "$f" | awk -v OFS='\t' -v S="$sample" '
    $4 ~ /:MainPeak$/ {
      mid = int(($2+$3)/2)
      print $1, mid, mid+1, S
    }' >> summits.bed
done

# sort for bedtools
sort -k1,1 -k2,2n summits.bed > summits.sorted.bed

### Cluster summits across files with a ≥50% overlap rule (BEDTools/2.31.0-GCC-12.3.0)
For ~70 bp peaks, “≥50% overlap” equals summits within 35 bp

bedtools cluster -d 35 -i summits.sorted.bed > summits.clustered.bed
# Columns now: chr  start  end  sample  clusterID

### For each cluster: support, median summit, SD, and IQR
Compute:
* **support** = # of **unique samples** contributing a summit in the cluster
* **median** summit position (consensus 1-bp peak)
* **SD** and **IQR** of summit positions (positional tightness)

LC_ALL=C ./summarize_clusters.sh summits.clustered.bed > clusters_summary.bed

Output columns (`clusters_summary.bed`):
chr   median_start   median_end(=start+1)   cluster_id   support   sd_start   iqr_start   n

### Take top 1,000 peak clusters with support (unique experiments) >100 based on smallest stdev (tightest)

awk '$5 > 100' clusters_summary.bed | awk '$1 != "mitochondria"' | sort -k6,6n | head -1000 | awk '{print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $2}' | sort -k1,1 -k2,2n > NucMap_iNPS_1000.bed

### Append Triton-style header (manually):
chrom	chromStart	chromEnd	name	position