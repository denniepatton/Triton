# Triton.snakefile
# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v0.3.1, 04/23/2024

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
        fm_file = (config['results_dir']+"/{samples}_TritonFeatures.tsv"),
        signal_file = (config['results_dir']+"/{samples}_TritonProfiles.npz")
    params:
        sample_name = "{samples}",
        annotation = config['annotation'],
        reference_genome = config['reference_genome'],
        results_dir = config['results_dir'],
        map_quality = config['map_quality'],
        size_range=config['size_range'],
        cpus = config['triton_main']['ncpus'],
        run_mode = config['run_mode'],
        bg_panel = config['bg_panel'],
        tumor_fraction = lambda wildcards: '--tumor_fraction ' + str(config["samples"][wildcards.samples]['tfx']) if 'tfx' in config["samples"][wildcards.samples] else ''
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
            --run_mode {params.run_mode} \
            --subtract_background_panel {params.bg_panel} \
            {params.tumor_fraction}
        """

rule combine_fms:
    input:
        fm_files = expand("{results_dir}/{samples}_TritonFeatures.tsv", results_dir=config['results_dir'], samples=config['samples'].keys())
    output:
        final = "{results_dir}/TritonCompositeFM.tsv".format(results_dir=config['results_dir'])
    params:
        results_dir=config['results_dir']
    shell:
        'python Triton/triton_cleanup.py --results_dir {params.results_dir}'

