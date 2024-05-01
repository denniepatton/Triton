# generate_panel.snakefile
# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v0.0.1, 04/12/2024

"""
# before running snakemake at Fred Hutch, do in tmux terminal:
ml snakemake/5.19.2-foss-2019b-Python-3.7.4
ml Python/3.7.4-foss-2019b-fh1

# command to run snakemake (remove -np at end when done validating):
snakemake -s generate_panel.snakefile --latency-wait 60 --keep-going --cluster-config config/cluster_slurm.yaml --cluster "sbatch -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -c {cluster.ncpus} -n {cluster.ntasks} -o {cluster.output} -J {cluster.JobName}" -j 40 -np
"""
import os

configfile: "config/HD_samples.yaml"
configfile: "config/config.yaml"
configfile: "config/cluster_slurm.yaml"

# Get the base name of the annotation file without the extension
stripped_annotation = os.path.splitext(os.path.basename(config['annotation']))[0]

rule all:
    input:
        expand("{results_dir}/{samples}_TritonRawPanel_{annotation}.npz", results_dir=config['results_dir'], samples=config['samples'].keys(), annotation=stripped_annotation),
        expand("{results_dir}/" + stripped_annotation + "_BackgroundPanel.npz", results_dir=config['results_dir'])

rule generate_panel:
    input:
        bam_path = lambda wildcards: config["samples"][wildcards.samples]['bam'],
        bias_path = lambda wildcards: config["samples"][wildcards.samples]['GC_bias']
    output:
        "{results_dir}/{samples}_TritonRawPanel_{annotation}.npz".format(results_dir=config['results_dir'], samples="{samples}", annotation=stripped_annotation)
        # temp("{results_dir}/{samples}_TritonRawPanel_{annotation}.npz".format(results_dir=config['results_dir'], samples="{samples}", annotation=stripped_annotation))
    params:
        sample_name = "{samples}",
        annotation = config['annotation'],
        reference_genome = config['reference_genome'],
        results_dir = config['results_dir'],
        map_quality = config['map_quality'],
        size_range=config['size_range'],
        cpus = config['generate_panel']['ncpus'],
        run_mode = config['run_mode']
    shell:
        "python ../Triton/Triton.py --sample_name {params.sample_name} \
        --input {input.bam_path} --bias {input.bias_path} \
        --annotation {params.annotation} \
        --reference_genome {params.reference_genome} \
        --results_dir {params.results_dir} --map_quality {params.map_quality} \
        --size_range {params.size_range} --cpus {params.cpus} \
        --frag_dict ../nc_info/NCDict.pkl \
        --run_mode {params.run_mode} --generate_panel "

rule triton_panel:
    input:
        expand("{results_dir}/{samples}_TritonRawPanel_{annotation}.npz", results_dir=config['results_dir'], samples=config['samples'].keys(), annotation=stripped_annotation)
    output:
        final = expand("{results_dir}/" + stripped_annotation + "_BackgroundPanel.npz", results_dir=config['results_dir'])
    params:
        results_dir=config['results_dir']
    shell:
        'python ../Triton/triton_panel.py --results_dir {params.results_dir}'
