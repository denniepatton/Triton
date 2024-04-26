MANE transctipt info was taken directly from the MANE.GRCh38.v1.3.summary.txt file, see https://www.ncbi.nlm.nih.gov/refseq/MANE/

TFBS sites were generated as follows:

Homo_sapiens_meta_clusters.interval was first downloaded from https://gtrd.biouml.org/downloads/19.10/chip-seq/Homo%20sapiens_meta_clusters.interval.gz
This file was then seperated into TF-specific files using the following command (N.B. the -t$'\t' or -F"\t" is NEEDED due to white space in some columns)
	awk -F"\t" '{gsub("/", "-", $6); {print > $6 ".txt"}}' Homo_sapiens_meta_clusters.interval
Files were BED-sorted and had header re-appended (header lost in separation above):
	for file in *.txt; do sort -t$'\t' -k1,1 -k2,2n ${file} > ${file/.txt/_sorted.txt}; done
	for file in *_sorted.txt; do sed '1s/^/#CHROM\tSTART\tEND\tsummit\tuniprotId\ttfTitle\tcell.set\ttreatment.set\texp.set\tpeak-caller.set\tpeak-caller.count\texp.count\tpeak.count\n/' ${file} > ${file/_sorted/}; done
	rm *_sorted.txt

Sites were then restricted to those with at least 3 peaks, then overlapping sites were merged:
	for file in *_raw.txt; do awk -F"\t" '$13 > 2' ${file} | bedtools merge -i - -c 4,5,6,7,8,9,10,11,12,13 -o mean,distinct,distinct,distinct,distinct,distinct,distinct,mean,mean,mean > ${file/raw/merged}; done
remove sites overlapping (by even 1bp) exclusions:
	for file in *_merged.txt; do bedtools intersect -v -a ${file} -b ../../Exclusions/all_exclusions_merged.bed > ${file/merged/filtered}; done
next find a window of 10,000 (+/-5,000) bp around the "position" index, taken as the int(mean(START+STOP)) and require that window to completely overlap contiguous regions with mappability > 0.90 (minimum, not mean)
	for file in *_filtered.txt; do awk -F"\t" '{print $1 "\t" $2 "\t" $3 "\t" $12 "\t" $13 "\t" int(($2+$3)/2)}' ${file} | awk -F"\t" '{print $1 "\t" $6-5000 "\t" $6+5000 "\t" $4 "\t" $5 "\t" $6 "\t" $2 "\t" $3}' | awk '$2 > 0' | bedtools intersect -wa -f 1 -a - -b ../../Exclusions/k100.Umap.MultiTrackMappability_.90.bed | awk -F"\t" '{print $1 "\t" $7 "\t" $8 "\t" $4 "\t" $5 "\t" $6}' | sort -k1,1 -k2,2n | sed '1s/^/chrom\tchromStart\tchromEnd\texpCount\tpeakCount\tposition\n/' > ${file/.txt/.bed}; done
	rm *_merged.txt
	rm *_filtered.txt
	mv *_filtered.bed ../GTRD_FILTERED/

Next restrict to sites with at least 10,000 sites
first count total lines in each site list:
	wc -l *_filtered.bed >> filtered_site_numbers.txt
then get list of tfbs to drop (10,001 with header):
	awk '$1 < 10002' filtered_site_numbers.txt | awk '{print $2}' > drop_tfbs.txt
remove those tfbss:
	xargs rm < drop_tfbs.txt
	rm filtered_site_numbers.txt
	rm drop_tfbs.txt

Next these sites were sorted and filtered to the top 10,000/1,000  based off of the columns "peakCount" > "expCount" and header re-applied
	for file in *_filtered.bed; do sort -t$'\t' -k5,5nr -k4,4nr ${file} | head -1000 | sort -t$'\t' -k1,1 -k2,2n | sed '1s/^/chrom\tchromStart\tchromEnd\texpCount\tpeakCount\tposition\n/' > ${file/filtered/top1000}; done
	for file in *_filtered.bed; do sort -t$'\t' -k5,5nr -k4,4nr ${file} | head -10000 | sort -t$'\t' -k1,1 -k2,2n | sed '1s/^/chrom\tchromStart\tchromEnd\texpCount\tpeakCount\tposition\n/' > ${file/filtered/top10000}; done
finally BEDs were re-organized
	mv GTRD_FILTERED/*_top10000.bed GTRD_F10000/
	mv GTRD_FILTERED/*_top1000.bed GTRD_F1000/

Lastly, produce lists of files for downstream use:
	for file in GTRD_F10000/*.bed; do realpath ${file} >> GTRD_F10000.tsv; done
	for file in GTRD_F1000/*.bed; do realpath ${file} >> GTRD_F1000.tsv; done

Recall that F = FILTERED = sites with at least two peaks, merged if touching, not overlapping exclusion regions, fully overlapping contiguous mappability>=0.90 regions, with top 10/1K sites taken based on peak.count then exp.count
N.B. that merged sites used MEAN peak.count and exp.count between pre-merged sites, produced by BEDTools
