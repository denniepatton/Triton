#!/usr/bin/env bash
# Usage: ./summarize_clusters.sh summits.clustered.bed > clusters_summary.bed
# Requires: gawk (GNU awk)

in="${1:-summits.clustered.bed}"

gawk -v OFS='\t' '
function flush_cluster(   n,i,mean,sd,q1,q3,med,k) {
  if (count==0) return

  # sort positions for median/IQR
  delete sorted
  n = asort(pos, sorted)

  # median (integer midpoint for even n)
  if (n % 2) { med = sorted[(n+1)/2] }
  else       { med = int((sorted[n/2] + sorted[n/2 + 1]) / 2) }

  # Tukey hinges for Q1/Q3 (simple, robust)
  q1 = sorted[int((n+3)/4)]
  q3 = sorted[int((3*n+1)/4)]
  iqr = q3 - q1

  # sample SD
  mean = sum / n
  if (n > 1) sd = sqrt((sumsq - sum*sum/n) / (n - 1)); else sd = 0

  # unique sample support
  sup = 0; for (k in seen) sup++

  # Output: chrom  median  median+1  cluster_name  support  sd  iqr  n
  print chr_cur, med, med+1, ("cluster_" cluster_id), sup, sd, iqr, n
}

BEGIN { FS = OFS = "\t" }

# Input columns: chr  start  end  sample  clusterID
NR == 1 {
  cluster_id = $5
  chr_cur    = $1
}

{
  chr   = $1
  start = $2 + 0
  samp  = $4
  cid   = $5

  # New cluster → flush previous
  if (cid != cluster_id) {
    flush_cluster()
    # reset accumulators for new cluster
    delete pos; delete seen
    count = 0; sum = 0; sumsq = 0
    cluster_id = cid
    chr_cur    = chr
  }

  # accumulate for current cluster
  pos[++count] = start
  sum   += start
  sumsq += start * start
  seen[samp] = 1
}

END {
  flush_cluster()
}
' "$in"
