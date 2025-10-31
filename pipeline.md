cd src

# 1) build PDX matrices
python preprocess_pdx1.py --overwrite

# 2) train aligner on PDX1 (example)
python train_aligner.py --target_dataset pdx1 --target_gex ../data/PDX1_gex.csv --target_info ../data/PDX1_info.csv --device cpu

# 3) export weights from trained aligner
python export_target_embeddings.py --checkpoint ckpts/THERAPI_aligner.pt --target_dataset pdx1 --output_prefix ../data/PDX1_align --device cpu

# 4) assemble predictor inputs (provide --treatment_map for extra aliases)
python build_target_predictor_inputs.py --align_prefix ../data/PDX1_align --output_prefix ../data/PDX1

# 5) score predictor ensemble
python test_PDX1.py --data_dir ../data/ --output_dir ../output/ --device cpu


scp -r data_pdx2 data_pdx1 data haeun@147.47.67.51:/data/project/haeun/THERAPI