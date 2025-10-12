rule create_dirs:
    output:
        brain_dir="{outdir}/{guid}/mri/orig"
    shell:
        """
        mkdir -p {output.brain_dir}
        """

rule copy_raw_image:
    input:
        raw=config["raw_mri_path"]
    output:
        "{outdir}/{guid}/mri/orig/001.mgz"
    shell:
        """
        cp {input.raw} {output}
        """

rule conform_image:
    input:
        "{outdir}/{guid}/mri/orig/001.mgz"
    output:
        "{outdir}/{guid}/mri/orig_conform.mgz"
    container:
        "singularity/freesurfer.sif"
    shell:
        """
        export FREESURFER_HOME=/usr/local/freesurfer
        export FS_LICENSE=/usr/local/freesurfer/.license
        mri_convert --conform {input} {output}
        """

rule bias_correction:
    input:
        "{outdir}/{guid}/mri/orig_conform.mgz"
    output:
        "{outdir}/{guid}/mri/orig_nu.mgz"
    container:
        "singularity/ants.sif"
    shell:
        """
        N4BiasFieldCorrection -i {input} -o {output}
        """

rule synthseg_segmentation:
    input:
        "{outdir}/{guid}/mri/orig_nu.mgz"
    output:
        "{outdir}/{guid}/mri/aparc+aseg.mgz"
    container:
        "singularity/freesurfer.sif"
    shell:
        """
        mri_synthseg --i {input} --o {output} --robust --cpu --threads 8 --parc
        """

rule talairach_transform:
    input:
        "{outdir}/{guid}/mri/orig_nu.mgz"
    output:
        "{outdir}/{guid}/mri/transforms/talairach.xfm.lta"
    container:
        "singularity/freesurfer.sif"
    shell:
        """
        mkdir -p {wildcards.outdir}/{wildcards.guid}/mri/transforms
        talairach_avi --i {input} --xfm {wildcards.outdir}/{wildcards.guid}/mri/transforms/talairach.auto.xfm
        lta_convert --src {input} --trg $FREESURFER_HOME/average/mni305.cor.mgz \
            --inxfm {wildcards.outdir}/{wildcards.guid}/mri/transforms/talairach.auto.xfm \
            --outlta {output}
        """

rule brain_extraction:
    input:
        "{outdir}/{guid}/mri/orig_nu.mgz"
    output:
        brain="{outdir}/{guid}/mri/brain.mgz",
        mask="{outdir}/{guid}/mri/mask.mgz"
    container:
        "singularity/freesurfer.sif"
    shell:
        """
        mri_synthstrip --i {input} --o {output.brain} --mask {output.mask}
        """

rule brain_vol2vol:
    input:
        brain="{outdir}/{guid}/mri/brain.mgz",
        xfm="{outdir}/{guid}/mri/transforms/talairach.xfm.lta"
    output:
        "{outdir}/{guid}/mri/brain_talairach.nii.gz"
    container:
        "singularity/freesurfer.sif"
    shell:
        """
        mri_vol2vol --mov {input.brain} \
            --targ $FREESURFER_HOME/average/mni305.cor.mgz \
            --reg {input.xfm} \
            --o {output}
        """

rule seg_vol2vol:
    input:
        seg="{outdir}/{guid}/mri/aparc+aseg.mgz",
        xfm="{outdir}/{guid}/mri/transforms/talairach.xfm.lta"
    output:
        "{outdir}/{guid}/mri/aparc+aseg_talairach.nii.gz"
    container:
        "singularity/freesurfer.sif"
    shell:
        """
        mri_vol2vol --mov {input.seg} \
            --targ $FREESURFER_HOME/average/mni305.cor.mgz \
            --reg {input.xfm} \
            --o {output}
        """
