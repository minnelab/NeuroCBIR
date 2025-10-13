rule run_neurocbir_whole_brain:
    input:
        brain="{outdir}/{guid}/mri/brain_talairach.nii.gz",
    output:
        result="{outdir}/{guid}/neurocbir_report/whole_brain/whole_brain.json"
    container:
        "singularity/neurocbir.sif"
    params:
        # emb_dataset_path=lambda w: config.get("emb_dataset_path", ""),
        # region=lambda w: config.get("region", ""),
        scope="whole_brain",
        emb_dataset_path=lambda w: config.get("emb_dataset_path_whole_brain", ""),
        top_k=lambda w: config.get("top_k", ""),
        device=lambda w: config.get("device", ""),
        o_path=lambda w: f"{w.outdir}/{w.guid}/neurocbir_report/whole_brain",
        user_config=lambda w: config.get("user_config", ""),
        internal_config=lambda w: config.get("internal_config", ""),
        
    shell:
        """
        cd /app
        python -m neurocbir \
        --brain_path {input.brain} \
        --scope {params.scope} \
        --emb_dataset_path {params.emb_dataset_path} \
        --top_k {params.top_k} \
        --device {params.device} \
        --o_path {params.o_path} 
        """

rule run_neurocbir_region:
    input:
        brain="{outdir}/{guid}/mri/brain_talairach.nii.gz",
        seg="{outdir}/{guid}/mri/aparc+aseg_talairach.nii.gz",
    output:
        result="{outdir}/{guid}/neurocbir_report/region/{region}.json"
    container:
        "singularity/neurocbir.sif"
    params:
        scope="region",
        emb_dataset_path=lambda w: config.get("emb_dataset_path_region", ""),
        top_k=lambda w: config.get("top_k", ""),
        device=lambda w: config.get("device", ""),
        o_path=lambda w: f"{w.outdir}/{w.guid}/neurocbir_report/region",
        user_config=lambda w: config.get("user_config", ""),
        internal_config=lambda w: config.get("internal_config", ""),
        
    shell:
        """
        cd /app
        python -m neurocbir \
        --brain_path {input.brain} \
        --seg_path {input.seg} \
        --scope {params.scope} \
        --region {wildcards.region} \
        --emb_dataset_path {params.emb_dataset_path} \
        --top_k {params.top_k} \
        --device {params.device} \
        --o_path {params.o_path}
        """
