# Hierarchical THERAPI - Quick Start Guide

## 🚀 TL;DR

Hierarchical THERAPI is a unified tissue-aware model for drug response prediction. One model replaces many tissue-specific models.

## 📋 Prerequisites

```bash
# Python 3.9+
conda create -n hierarchical_therapi python=3.9
conda activate hierarchical_therapi
pip install -r requirements.txt
```

## 📂 Data Setup

Place TRANSACT data in `data/` folder:
```
data/
├── mini_cancer_genes.csv
├── GDSC/
├── PDXE/
├── TCGA/
└── HMF/
```

## 🏃 Quick Commands

### Train
```bash
python train_hierarchical.py \
    --source GDSC \
    --target TCGA \
    --config configs/hierarchical_config.yaml \
    --device cuda:0
```

### Evaluate
```bash
python evaluate_hierarchical.py \
    --model_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt \
    --target TCGA \
    --output_dir output/
```

### Analyze
```bash
jupyter notebook notebooks/tissue_routing_analysis.ipynb
```

## 🔧 Configuration

Edit `configs/hierarchical_config.yaml`:

```yaml
model:
  latent_dim: 128
  routing_strategy: 'gumbel'

training:
  batch_size: 128
  learning_rate: 0.001
  n_epochs: 200
```

## 📊 Key Outputs

After training:
- `ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt` - Trained model
- `log/*.log` - Training logs

After evaluation:
- `output/tissue_routing_confusion_TCGA.csv` - Routing accuracy
- `output/tissue_routing_summary.csv` - Summary statistics

## 💡 Common Use Cases

### 1. Standard Training
```bash
python train_hierarchical.py --source GDSC --target TCGA
```

### 2. Ablation Study
```python
from models.ablation_models import create_ablation_model

model = create_ablation_model('flat', n_genes=1817)  # No hierarchy
```

### 3. Custom Routing Strategy
```yaml
# In config file
model:
  routing_strategy: 'sparsemax'  # Instead of 'gumbel'
```

### 4. Tissue Routing Analysis
```python
# In notebook
tissue_weights = model.tissue_router(patient_encoding)
top_tissue = tissue_weights.argmax(dim=-1)
```

## 🎯 Model Variants

| Variant | Command |
|---------|---------|
| Full Model | `create_ablation_model('full_model', ...)` |
| No Hierarchy | `create_ablation_model('flat', ...)` |
| Uniform Weights | `create_ablation_model('uniform_weights', ...)` |
| Hard Routing | `create_ablation_model('hard_routing', ...)` |

## 📈 Expected Results

- **Routing Accuracy**: >70%
- **Inference Speedup**: 20-50%
- **Routing Entropy**: ~2-3 (decisive)
- **Temperature**: Converges to ~0.5

## 🐛 Troubleshooting

### Out of Memory
```yaml
training:
  batch_size: 64  # Reduce from 128
```

### Poor Routing
```yaml
training:
  loss_weights:
    routing: 0.8  # Increase from 0.4
```

### Slow Training
```bash
python train_hierarchical.py --device cuda:0  # Use GPU
```

## 📚 Documentation

- Full docs: [HIERARCHICAL_README.md](HIERARCHICAL_README.md)
- Implementation: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- Plan: [plan.md](plan.md)

## 🔍 File Reference

| File | Purpose |
|------|---------|
| `models/hierarchical_therapi.py` | Main model |
| `models/ablation_models.py` | Ablation variants |
| `utils/data_loader.py` | Data loading |
| `utils/tissue_mapping.py` | Tissue standardization |
| `train_hierarchical.py` | Training script |
| `evaluate_hierarchical.py` | Evaluation script |
| `configs/hierarchical_config.yaml` | Configuration |

## ⚡ Performance Tips

1. **Use GPU**: `--device cuda:0`
2. **Batch size**: Start with 128, adjust for memory
3. **Early stopping**: Monitor validation loss
4. **PDX validation**: Use PDX for hyperparameter tuning

## 🎓 Learning Path

1. Read [HIERARCHICAL_README.md](HIERARCHICAL_README.md)
2. Check [plan.md](plan.md) for design decisions
3. Run training on small dataset
4. Analyze results in notebook
5. Run ablation studies
6. Compare to baselines

## 🤝 Contributing

See main [README.md](README.md) for contribution guidelines.

## 📞 Support

- Issues: GitHub Issues
- Original THERAPI: inyoung.sung@snu.ac.kr

---

**Ready to go!** Start with `python train_hierarchical.py --help`
