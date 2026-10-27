# !/bin/bash

# Run command: bash ./tmp/run_ablation.sh


# echo "Running CBIR evaluation for eval__d_32__m_02__lid_0__mid_02__pn_005__lsn_02"
# python -m dev.scripts.whole_brain.train_contrastive_model --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_0__mid_02__pn_005__lsn_02/configs/train_contrastive_model.py
# python -m dev.scripts.whole_brain.run_cl_embedding --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_0__mid_02__pn_005__lsn_02/configs/run_cl_embedding.py
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_0__mid_02__pn_005__lsn_02/configs/run_cbir_eval.py
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_0__mid_02__pn_005__lsn_02/configs/run_sim_eval.py


# echo "Running CBIR evaluation for eval__d_32__m_02__lid_02__mid_02__pn_005__lsn_0"
# python -m dev.scripts.whole_brain.train_contrastive_model --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_02__mid_02__pn_005__lsn_0/configs/train_contrastive_model.py
# python -m dev.scripts.whole_brain.run_cl_embedding --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_02__mid_02__pn_005__lsn_0/configs/run_cl_embedding.py
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_02__mid_02__pn_005__lsn_0/configs/run_cbir_eval.py
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_02__mid_02__pn_005__lsn_0/configs/run_sim_eval.py

# echo "Running CBIR evaluation for eval__d_32__ahn_02__lsn_0__asn_02__pn_005__lhn_0"
# python -m dev.scripts.whole_brain.train_contrastive_model --config dev/data_private/results/whole_brain/eval__d_32__ahn_02__lsn_0__asn_02__pn_005__lhn_0/configs/train_contrastive_model.py
# python -m dev.scripts.whole_brain.run_cl_embedding --config dev/data_private/results/whole_brain/eval__d_32__ahn_02__lsn_0__asn_02__pn_005__lhn_0/configs/run_cl_embedding.py
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_32__ahn_02__lsn_0__asn_02__pn_005__lhn_0/configs/run_cbir_eval.py
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_32__ahn_02__lsn_0__asn_02__pn_005__lhn_0/configs/run_sim_eval.py

# echo "Running ablation experiments for whole brain evaluation..."

# # d = 16
# echo "Running CBIR evaluation for d=16, p=30, m=0.5..."
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_16__m_05__lid_1/configs/run_cbir_eval.py
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_16__m_05__lid_1/configs/run_sim_eval.py

# echo "Running CBIR evaluation for d=16, p=30, m=0.2..."
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_16__m_02__lid_1/configs/run_cbir_eval.py
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_16__m_02__lid_1/configs/run_sim_eval.py

# # d = 32
# echo "Running CBIR evaluation for d=32, m=0.5, p=30..."
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_32__m_05__lid_1/configs/run_cbir_eval.py
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_32__m_05__lid_1/configs/run_sim_eval.py

# echo "Running CBIR evaluation for d=32, m=0.5, p=30..."
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_1/configs/run_cbir_eval.py
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_1/configs/run_sim_eval.py

# echo "Running CBIR evaluation for d=32, m=0.2, p=30..."
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_02/configs/run_cbir_eval.py
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_02/configs/run_sim_eval.py

# echo "Running CBIR evaluation for d=32, m=0.2, p=30, lid=0.2, mid=0.1..."
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_02__mid_01/configs/run_sim_eval.py
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_02__mid_01/configs/run_cbir_eval.py

# echo "Running CBIR evaluation for d=32, m=0.2, p=30, lid=0.2, mid=0.1, pn=0.05..."
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_02__mid_01__pn_005/configs/run_sim_eval.py
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_02__mid_01__pn_005/configs/run_cbir_eval.py

# echo "Running CBIR evaluation for d=32, m=0.2, p=30, lid=0.2, mid=0.2, pn=0.05..."
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_02__mid_02__pn_005/configs/run_sim_eval.py
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_02__mid_02__pn_005/configs/run_cbir_eval.py
# python -m dev.scripts.whole_brain.metrics_json_to_csv --config dev/data_private/results/whole_brain/eval__d_32__m_02__lid_02__mid_02__pn_005/configs/metrics_json_to_csv_config.py

# # d = 64
# echo "Running CBIR evaluation for d=64, m=0.5, p=30..." 
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_64__m_05__lid_1/configs/run_cbir_eval.py
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_64__m_05__lid_1/configs/run_sim_eval.py

# echo "Running CBIR evaluation for d=64, m=0.2, p=30, lid=0.2, mid=0.1, pn=0.05..."
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_64__m_02__lid_02__mid_01__pn_005/configs/run_cbir_eval.py
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_64__m_02__lid_02__mid_01__pn_005/configs/run_sim_eval.py

# # d = 128
# echo "Running CBIR evaluation for d=128, m=0.5, p=30..." 
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_128__m_05__lid_1/configs/run_sim_eval.py
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_128__m_05__lid_1/configs/run_cbir_eval.py

# # d = 256
# echo "Running CBIR evaluation for d=256, m=0.5, p=30..." 
# python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__d_256__m_05__lid_1/configs/run_sim_eval.py
# python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__d_256__m_05__lid_1/configs/run_cbir_eval.py

# flatten VAE embedding
echo "Flattening VAE embedding"
python -m dev.scripts.whole_brain.run_flatten_vae_embedding --config dev/data_private/results/whole_brain/eval__vae_emb/configs/flatten_vae_embedding.py
python -m dev.scripts.whole_brain.run_cbir_eval --config dev/data_private/results/whole_brain/eval__vae_emb/configs/run_cbir_eval.py
python -m dev.scripts.whole_brain.run_sim_eval --config dev/data_private/results/whole_brain/eval__vae_emb/configs/run_sim_eval.py


# echo "Finished running ablation experiments for whole brain evaluation."


# Region brain
# echo "Preparing MS-SSIM dataset for region brain evaluation..."
# python -m dev.scripts.region_brain.prepare_mssim_ds

# echo "Running ablation experiments for region brain evaluation..."

# echo "Running CBIR evaluation for d=32, m=0.2, p=30, lid=0.2, mid=0.2, pn=0.05..."
# python -m dev.scripts.region_brain.run_sim_eval --config dev/data_private/results/region_brain/eval__d_32__m_02__lid_02__mid_02__pn_005/configs/run_sim_eval.py
# python -m dev.scripts.region_brain.run_cbir_eval --config dev/data_private/results/region_brain/eval__d_32__m_02__lid_02__mid_02__pn_005/configs/run_cbir_eval.py
# python -m dev.scripts.region_brain.metrics_json_to_csv --config dev/data_private/results/region_brain/eval__d_32__m_02__lid_02__mid_02__pn_005/configs/metrics_json_to_csv_config.py


# Methods comparison
# echo "cnn_pretrained_model"
# python -m dev.scripts.whole_brain.run_sim_eval --config data/multi_comp/cnn_pretrained_model/config/run_sim_eval.py
# python -m dev.scripts.whole_brain.run_cbir_eval --config data/multi_comp/cnn_pretrained_model/config/cbir_eval_config.py

# echo "large_scale_medical"
# python -m dev.scripts.whole_brain.run_sim_eval --config data/multi_comp/large_scale_medical/config/run_sim_eval.py
# python -m dev.scripts.whole_brain.run_cbir_eval --config data/multi_comp/large_scale_medical/config/cbir_eval_config.py

# echo "resnet10"
# python -m dev.scripts.whole_brain.run_sim_eval --config data/multi_comp/resnet10/config/run_sim_eval.py
# python -m dev.scripts.whole_brain.run_cbir_eval --config data/multi_comp/resnet10/config/cbir_eval_config.py

# echo "resnet50"
# python -m dev.scripts.whole_brain.run_sim_eval --config data/multi_comp/resnet50/config/run_sim_eval.py
# python -m dev.scripts.whole_brain.run_cbir_eval --config data/multi_comp/resnet50/config/cbir_eval_config.py

# echo "resnet101"
# python -m dev.scripts.whole_brain.run_sim_eval --config data/multi_comp/resnet101/config/run_sim_eval.py
# python -m dev.scripts.whole_brain.run_cbir_eval --config data/multi_comp/resnet101/config/cbir_eval_config.py

# scancel $SLURM_JOB_ID

