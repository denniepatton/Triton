# Triton.snakefile
# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v2.0, 09/16/2025

"""
# before running snakemake at Fred Hutch, do in tmux terminal:
ml snakemake/5.19.2-foss-2019b-Python-3.7.4
ml Python/3.7.4-foss-2019b-fh1

# command to run snakemake (remove -np at end when done validating):
snakemake -s Triton.snakefile --latency-wait 60 --keep-going --cluster-config config/cluster_slurm.yaml --cluster "sbatch -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -c {cluster.ncpus} -n {cluster.ntasks} -o {cluster.output} -J {cluster.JobName}" -j 40 -np
"""

configfile: "config/samples.yaml"
configfile: "config/config.yaml"
configfile: "config/cluster_slurm.yaml"

rule all:
	input:
		expand("{results_dir}/TritonCompositeFM.tsv",results_dir=config['results_dir'])

rule triton_main:
    input:
        bam_path = lambda wildcards: config["samples"][wildcards.samples]['bam'],
        bias_path = lambda wildcards: config["samples"][wildcards.samples]['GC_bias']
    output:
        fm_file = (config['results_dir']+"/{samples}/{samples}_TritonFeatures.tsv"),
        signals_file = (config['results_dir']+"/{samples}/{samples}_TritonSignalProfiles.npz"),
        fragments_file = (config['results_dir']+"/{samples}/{samples}_TritonFragmentationProfiles.npz")
    params:
        sample_name = "{samples}",
        annotation = config['annotation'],
        reference_genome = config['reference_genome'],
        results_dir = config['results_dir'],
        map_quality = config['map_quality'],
        size_range=config['size_range'],
        cpus = config['triton_main']['ncpus'],
        run_mode = config['run_mode'],
    shell:
        """
        python Triton/Triton.py \
            --sample_name {wildcards.samples} \
            --input {input.bam_path} \
            --bias {input.bias_path} \
            --annotation {params.annotation} \
            --reference_genome {params.reference_genome} \
            --results_dir {params.results_dir} \
            --map_quality {params.map_quality} \
            --size_range {params.size_range} \
            --cpus {params.cpus} \
            --run_mode {params.run_mode}
        """

rule combine_fms:
    input:
        fm_files = expand("{results_dir}/{samples}/{samples}_TritonFeatures.tsv", results_dir=config['results_dir'], samples=config['samples'].keys())
    output:
        final = "{results_dir}/TritonCompositeFM.tsv".format(results_dir=config['results_dir'])
    params:
        results_dir=config['results_dir']
    shell:
        'python Triton/triton_cleanup.py --results_dir {params.results_dir}'

