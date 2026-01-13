# nc_dist.snakefile
# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v0.0.1, 04/04/2024

"""
# before running snakemake at Fred Hutch, do in tmux terminal:
ml snakemake/5.19.2-foss-2019b-Python-3.7.4
ml Python/3.7.4-foss-2019b-fh1

# command to run snakemake (remove -np at end when done validating):
snakemake -s nc_dist.snakefile --latency-wait 60 --keep-going --cluster-config config/cluster_slurm.yaml --cluster "sbatch -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -c {cluster.ncpus} -n {cluster.ntasks} -o {cluster.output} -J {cluster.JobName}" -j 40 -np
"""

configfile: "config/samples.yaml"
configfile: "config/config.yaml"
configfile: "config/cluster_slurm.yaml"

rule all:
    input:
        expand(config['results_dir']+"/{sample}_TritonNucPlacementProfiles.npz", sample=config["samples"].keys())

rule nc_dist:
	input:
		bam_path = lambda wildcards: config["samples"][wildcards.sample]["bam"]
	output:
		disp_file = (config['results_dir']+"/{sample}_TritonNucPlacementProfiles.npz")
	params:
		sample_name = "{sample}",
		annotation = config['annotation'],
		results_dir = config['results_dir'],
		map_quality = config['map_quality'],
		size_range=config['size_range']
	shell:
		"python ../Triton/nc_dist.py --sample_name {params.sample_name} \
		--input {input.bam_path} --annotation {params.annotation} \
		--results_dir {params.results_dir} --map_quality {params.map_quality} \
		--size_range {params.size_range}"
