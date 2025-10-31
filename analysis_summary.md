# THERAPI repository architecture and pipeline overview

## Project structure
- `src/`: core training and inference scripts plus model definitions (autoencoder-based aligner, domain classifiers, response predictor). Includes utilities (`utils.py`) for logging, seeding, early stopping, and a center-loss implementation.
- `data/`: preprocessed matrices from GDSC (cell line expression, perturbations, response labels) and TCGA (patient expression, perturbations, labels). Contains cross-validation index splits `GDSC_split/fold_*_indices.pkl` for predictor training.
- `data_pdx1/` and `data_pdx2/`: supplementary patient-derived xenograft datasets (copy number, RNAseq, drug response curves) used in extended experiments.
- `img/`: figure assets (e.g., `Overview.png` matches README architecture).
- `log/`: archived training logs.
- `ckpts/`: saved model checkpoints created by training scripts.

## Data inputs
- **GDSC gene expression** (`GDSC_gex.csv`, 673 cell lines × 979 genes) with corresponding metadata (`GDSC_info.csv`).
- **TCGA gene expression** labeled (`TCGA_labeled_gex.csv`, 358 patients) and unlabeled (`TCGA_unlabeled_gex.csv`, 8042 patients); metadata ties samples to tissues.
- **Perturbation embeddings**: `GDSC_perturbation.npy` and `TCGA_perturbatio_float16.npy` (drug-induced transcriptomic signatures, 978-dim) plus `*_perturbation_compound*.npy` with chemical descriptors. PDX processing scripts produce analogous tensors (`PDX1_perturbation.npy`, `PDX1_perturbation_compound.npy`).
- **Rank representations**: `GDSC_rankrepresentation.csv`, `TCGA_rankrepresentation.csv`, and (after preprocessing) `PDX1_rankrepresentation.csv` (256 features) summarizing expression for use with predictor.
- **Drug response labels**: `GDSC_resp.csv` (IC50, pIC50, binary label) and `TCGA_resp.csv` (clinical response labels for evaluation).
- **PDX1 processed inputs** (generated on demand): `PDX1_gex.csv` (log-transformed expression aligned to GDSC genes), `PDX1_info.csv` (tumor type metadata harmonized with GDSC tissue labels), rank and perturbation tensors, and response table `PDX1_resp.csv` derived from PDX curve metrics.
- **Cross-validation**: ten pickled index splits for training/validation/testing on GDSC response data.

## Model architecture
THERAPI operates in two stages:

1. **Cell line–patient alignment (train_aligner.py)**
   - **GDSC autoencoder (GDSC_AE)** learns latent embeddings `z` from cell-line expression with reconstruction objective and center loss for tissue clustering, mirroring the $E_S$/$D_S$ encoders in the paper.
   - **TCGA weight encoder (TCGA_weightencoder)** applies the attention mechanism described in Equation (1) of the paper—querying patient embeddings (`Q z_t`) against projected cell-line embeddings (`K z^s_i`) to derive heterogeneity-aware weights $w_i^{(t,s)}$ and form a weighted latent mixture.
   - **Domain classifiers (Emb_Dis_classifier, Exp_Dis_classifier)** enforce tissue consistency via cross-entropy loss on both latent and reconstructed spaces, matching the dual latent/reconstruction classifiers noted in Section 2 of the manuscript.
   - Loss terms correspond to the reconstruction loss $\mathcal{L}_{rec}$, tissue-informed cross-entropy losses, and center loss $\mathcal{L}_{center}$ for compact tissue clusters, as summarized in Table 2 (ablation) of the paper.

2. **Drug response prediction (train_predictor.py / test_TCGA.py)**
   - **Dataset**: `ExpDrugDataset` pairs per-drug perturbation embeddings, rank features, chemical fingerprints, and response labels.
   - **Network (Response_predictor)**: three parallel MLP branches encode perturbation, gene feature, and chemical vectors; concatenated features flow through two dense layers to output drug response logit—aligned with the paper’s perturbation + rank feature integration (Table 2 ablations show both components are critical).
   - **Training**: 10-fold cross-validation on GDSC dataset with BCEWithLogits loss, Adam optimizer, batch size 512, early stopping based on validation loss. Models stored as `THERAPI_predictor_CV*.pt`.
   - **Inference**: `test_TCGA.py` loads trained folds, applies to TCGA perturbation/gene/chemical features, and reports metrics (AUC, AUPRC, accuracy, precision, F1). Averaged results saved in `output/THERAPI_test_TCGA.csv`.

## Pipeline summary
1. **Aligners** use unlabeled TCGA expression plus GDSC expression to learn shared latent space bridging preclinical (cell line) and clinical (patient) domains.
2. **Predictor** learns drug response mapping purely on preclinical (GDSC) domain using aligned representations (perturbation embeddings + rank features + chemical descriptors).
3. **Evaluation** applies the predictor to TCGA patient-derived perturbation features, exploiting learned alignments to estimate patient drug response likelihoods.

### PDX1 adaptation workflow
The codebase now supports end-to-end generation of PDX1 features and inference:

1. **Preprocess expression/metadata** – `preprocess_pdx1.py` ingests raw PDX RNAseq and response tables, harmonizes gene order with GDSC, maps tumor types onto GDSC tissue labels, fits a ridge-based surrogate to reproduce GDSC rank representations, and exports `PDX1_gex.csv`, `PDX1_info.csv`, `PDX1_rankrepresentation.csv`, plus a reusable rank-transform model (`pdx1_rank_transform.joblib`).
2. **Train aligner on PDX1** – `train_aligner.py` now accepts `--target_dataset`, `--target_gex`, and `--target_info` so the TCGA branch can be swapped for PDX1 (`python train_aligner.py --target_dataset pdx1 --target_gex ../data/PDX1_gex.csv --target_info ../data/PDX1_info.csv`). The aligner automatically remaps tissue labels across source and target domains.
3. **Export heterogeneity weights** – `export_target_embeddings.py` loads the trained checkpoint and emits per-sample attention weights, latent embeddings, and reconstructed expression (`PDX1_align_*` artifacts) required for downstream feature aggregation.
4. **Assemble predictor features** – `build_target_predictor_inputs.py` combines aligner weights with GDSC perturbation/chemical signatures to produce weighted drug embeddings, aligns rank features, and derives binary labels from PDX response categories. Outputs include `PDX1_perturbation.npy`, `PDX1_perturbation_compound.npy`, `PDX1_rank.npy`, `PDX1_samples.npy`, and `PDX1_resp.csv`, with skipped treatments logged for transparency.
5. **Inference & evaluation** – `test_PDX1.py` mirrors the TCGA evaluator, loading the generated tensors and pre-trained predictor ensemble to report AUROC/AUPRC/ACC/Precision/F1 plus per-sample probabilities (`output/THERAPI_test_PDX1_*.csv`).

> **Limitations**: only drugs present in GDSC (or user-specified via `--treatment_map`) can be embedded; combination therapies or novel agents require manual descriptors. The rank surrogate assumes GDSC rank features are well-approximated by ridge regression on log-transformed expression.

## Relation to paper
- The workshop paper “Transferring Preclinical Drug Response to Patient via Tumor Heterogeneity-aware Alignment and Perturbation Modeling” (MLGenX 2025) introduces THERAPI’s attention-based alignment. The repository’s `TCGA_weightencoder` mirrors the attention formula $w_i^{(t,s)} = \mathrm{softmax}(\langle Q z_t, K z_i^s \rangle)$ by projecting patient and cell-line latents before computing weights.
- Reconstruction loss, center loss, and tissue classification losses implemented in `train_aligner.py` reflect the paper’s combined objective (Section 2, Eq. 2) and ablation findings showing each term’s contribution (Table 2).
- Reported baseline comparison (Table 1) situates THERAPI’s AUROC (0.775 ± 0.034) and AUPRC (0.710 ± 0.024) above DA-based baselines (e.g., PANCDR), consistent with the performance targets for this repo’s predictor outputs.
- External validation on PDX cohorts noted in the paper is supported by the supplementary `data_pdx*` directories in the repo, which supply RNA-seq, copy-number, and drug response metrics for out-of-domain evaluation.

## Notes
- GPUs assumed (`cuda:1` default), but scripts can be run on CPU by adjusting `--device` argument.
- Ensure Python environment includes required packages from `requirements.txt` before running training scripts.
- New scripts rely on `scikit-learn` (already listed) and `joblib` (pulled transitively) for preprocessing and ridge modeling; rerun `pip install -r requirements.txt` if missing.
- When producing PDX1 artifacts, inspect `data/PDX1_preprocess_summary.json` and `data/PDX1_skipped.json` to audit dropped models or treatments before downstream training/testing.
