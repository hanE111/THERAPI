# Hierarchical THERAPI - Documentation Index

**Quick Navigation Guide for All Documentation**

---

## 🚀 Getting Started

**New to Hierarchical THERAPI? Start here:**

1. **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - 📋 Overview of entire implementation
2. **[QUICK_START.md](QUICK_START.md)** - ⚡ Quick commands and examples
3. **[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)** - 📖 Step-by-step complete guide

---

## 📚 Main Documentation

### For Users

**[HIERARCHICAL_README.md](HIERARCHICAL_README.md)** - 📘 Complete user documentation
- Installation instructions
- Usage examples
- Configuration guide
- Evaluation metrics
- Troubleshooting

**[THERAPI_COMPARISON.md](THERAPI_COMPARISON.md)** - 🔄 Original vs Hierarchical
- Why src/ folder wasn't modified
- Architecture comparison
- When to use each version
- Side-by-side workflow

### For Developers

**[ARCHITECTURE.md](ARCHITECTURE.md)** - 🏗️ Technical architecture
- Model diagrams
- Component details
- Data flow
- Implementation highlights

**[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - 🔬 Technical report
- What was implemented
- Validation against plan
- Code quality assessment
- Future extensions

---

## 🎯 By Use Case

### "I want to train the model"
1. Read: [QUICK_START.md](QUICK_START.md)
2. Follow: [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) Section 3
3. Run: `./run_hierarchical_pipeline.sh`

### "I want to understand the architecture"
1. Read: [ARCHITECTURE.md](ARCHITECTURE.md)
2. Review: [HIERARCHICAL_README.md](HIERARCHICAL_README.md) Architecture section
3. Study: `models/hierarchical_therapi.py`

### "I want to compare with original THERAPI"
1. Read: [THERAPI_COMPARISON.md](THERAPI_COMPARISON.md)
2. Run: Side-by-side workflow in comparison doc
3. Analyze: Results in `output/`

### "I want to run ablation studies"
1. Read: [HIERARCHICAL_README.md](HIERARCHICAL_README.md) Ablation section
2. Study: `models/ablation_models.py`
3. Create: Custom ablation script

### "I need to troubleshoot"
1. Check: [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) Section 7
2. Review: [QUICK_START.md](QUICK_START.md) Troubleshooting
3. Verify: Data format and paths

### "I want to analyze results"
1. Open: `notebooks/tissue_routing_analysis.ipynb`
2. Review: [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) Section 5
3. Generate: Custom analysis scripts

---

## 📋 Documentation Files

### Overview & Getting Started
- **[INDEX.md](INDEX.md)** ← You are here
- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - Complete implementation overview
- **[QUICK_START.md](QUICK_START.md)** - Quick reference guide
- **[README.md](README.md)** - Main repository README

### Comprehensive Guides
- **[HIERARCHICAL_README.md](HIERARCHICAL_README.md)** - Complete documentation
- **[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)** - Step-by-step workflow

### Technical Details
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Model architecture
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical report
- **[THERAPI_COMPARISON.md](THERAPI_COMPARISON.md)** - Original vs Hierarchical

### Planning & Design
- **[plan.md](plan.md)** - Original implementation plan

---

## 💻 Code Files

### Models (`models/`)
```python
hierarchical_therapi.py    # Main model: TissueRouter, HierarchicalAttention
ablation_models.py         # 8 ablation variants
```

### Utilities (`utils/`)
```python
data_loader.py            # TRANSACT data loading
tissue_mapping.py         # Tissue standardization (24 groups)
```

### Training Scripts
```bash
train_hierarchical.py              # Step 1: Aligner
train_hierarchical_predictor.py    # Step 2: Predictor
test_hierarchical_TCGA.py          # Step 3: Testing
evaluate_hierarchical.py           # Advanced evaluation
run_hierarchical_pipeline.sh       # Complete pipeline
```

### Configuration
```yaml
configs/hierarchical_config.yaml   # Model & training config
```

### Analysis
```jupyter
notebooks/tissue_routing_analysis.ipynb  # Interactive analysis
```

### Original THERAPI (`src/` - unchanged)
```python
model.py                  # Original models
utils.py                  # Utilities (used by hierarchical)
center_loss.py            # Loss function (used by hierarchical)
train_aligner.py          # Original training
train_predictor.py        # Original predictor
test_TCGA.py              # Original testing
```

---

## 🔍 Find What You Need

### By Topic

#### Installation & Setup
- Prerequisites: [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) Section 1
- Dependencies: [requirements.txt](requirements.txt)
- Environment: [HIERARCHICAL_README.md](HIERARCHICAL_README.md) Installation

#### Data
- Data structure: [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) Section 2
- Data loading: `utils/data_loader.py`
- TRANSACT format: [plan.md](plan.md) Data Structure

#### Training
- Quick train: [QUICK_START.md](QUICK_START.md) Train section
- Detailed train: [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) Section 3
- Configuration: `configs/hierarchical_config.yaml`

#### Evaluation
- Standard eval: [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) Section 4
- TRANSACT eval: [HIERARCHICAL_README.md](HIERARCHICAL_README.md) Evaluation
- Metrics: [ARCHITECTURE.md](ARCHITECTURE.md) Evaluation Metrics

#### Analysis
- Notebook: `notebooks/tissue_routing_analysis.ipynb`
- Tissue routing: [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) Section 5
- Interpretability: [HIERARCHICAL_README.md](HIERARCHICAL_README.md) Interpretability

#### Comparison
- Original vs Hierarchical: [THERAPI_COMPARISON.md](THERAPI_COMPARISON.md)
- When to use: [THERAPI_COMPARISON.md](THERAPI_COMPARISON.md) Section "When to Use"
- Side-by-side: [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) Section 6

### By Question

**"How do I...?"**
- ...install? → [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) Section 1
- ...train? → [QUICK_START.md](QUICK_START.md) or `./run_hierarchical_pipeline.sh`
- ...evaluate? → [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) Section 4
- ...analyze? → [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) Section 5
- ...compare with original? → [THERAPI_COMPARISON.md](THERAPI_COMPARISON.md)
- ...run ablations? → [HIERARCHICAL_README.md](HIERARCHICAL_README.md) Ablations

**"What is...?"**
- ...tissue routing? → [ARCHITECTURE.md](ARCHITECTURE.md) TissueRouter
- ...hierarchical attention? → [ARCHITECTURE.md](ARCHITECTURE.md) HierarchicalAttention
- ...Gumbel-Softmax? → [ARCHITECTURE.md](ARCHITECTURE.md) Technical Highlights
- ...TRANSACT? → [plan.md](plan.md) Context section

**"Why...?"**
- ...not modify src/? → [THERAPI_COMPARISON.md](THERAPI_COMPARISON.md)
- ...hierarchical? → [ARCHITECTURE.md](ARCHITECTURE.md) Overview
- ...24 tissue groups? → `utils/tissue_mapping.py` TISSUE_GROUPS

**"Where is...?"**
- ...the model code? → `models/hierarchical_therapi.py`
- ...data loading? → `utils/data_loader.py`
- ...configuration? → `configs/hierarchical_config.yaml`
- ...training script? → `train_hierarchical.py`
- ...ablations? → `models/ablation_models.py`

---

## 📊 Quick Command Reference

### Train Everything
```bash
./run_hierarchical_pipeline.sh
```

### Train Individual Steps
```bash
# Step 1
python train_hierarchical.py --source GDSC --target TCGA

# Step 2
python train_hierarchical_predictor.py \
    --aligner_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt

# Step 3
python test_hierarchical_TCGA.py \
    --aligner_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt
```

### Evaluate
```bash
python evaluate_hierarchical.py \
    --model_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt \
    --target TCGA
```

### Analyze
```bash
jupyter notebook notebooks/tissue_routing_analysis.ipynb
```

---

## 🎓 Learning Path

### Beginner (New to Project)
1. [FINAL_SUMMARY.md](FINAL_SUMMARY.md) - Get overview
2. [QUICK_START.md](QUICK_START.md) - Try basic commands
3. [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) - Follow complete workflow

### Intermediate (Ready to Train)
1. [HIERARCHICAL_README.md](HIERARCHICAL_README.md) - Read full docs
2. [THERAPI_COMPARISON.md](THERAPI_COMPARISON.md) - Understand differences
3. Run `./run_hierarchical_pipeline.sh`
4. Analyze results in notebook

### Advanced (Customization)
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Deep technical understanding
2. [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementation details
3. Study source code in `models/` and `utils/`
4. Create custom ablations and extensions

---

## 🔗 External Resources

### Original THERAPI
- GitHub: https://github.com/Sunginyoung/THERAPI
- Contact: inyoung.sung@snu.ac.kr

### TRANSACT Benchmark
- See [plan.md](plan.md) for data structure

### Dependencies
- PyTorch: https://pytorch.org/
- See [requirements.txt](requirements.txt) for versions

---

## ✅ Quick Checklist

Before training:
- [ ] Read [QUICK_START.md](QUICK_START.md)
- [ ] Install dependencies ([requirements.txt](requirements.txt))
- [ ] Download TRANSACT data (see [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md))
- [ ] Verify data structure (see [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) Section 2)

After training:
- [ ] Check logs in `log/`
- [ ] Verify checkpoints in `ckpts/`
- [ ] Review results in `output/`
- [ ] Run analysis notebook

For publication:
- [ ] Compare with original THERAPI ([THERAPI_COMPARISON.md](THERAPI_COMPARISON.md))
- [ ] Run all ablations ([HIERARCHICAL_README.md](HIERARCHICAL_README.md))
- [ ] Generate figures (notebook)
- [ ] Write methods using [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 💡 Tips

**For Quick Results:**
- Start with [QUICK_START.md](QUICK_START.md)
- Use `./run_hierarchical_pipeline.sh`
- Check `output/` for results

**For Understanding:**
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for visuals
- Study [THERAPI_COMPARISON.md](THERAPI_COMPARISON.md) for context
- Review source code with documentation

**For Troubleshooting:**
- Check [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) Section 7
- Verify data with validation script
- Review error messages in logs

**For Research:**
- Follow [plan.md](plan.md) for design rationale
- Use [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical depth
- Cite both original and hierarchical work

---

## 📞 Support

**Documentation Issues:**
- Review appropriate doc file above
- Check [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) troubleshooting

**Technical Issues:**
- GitHub Issues (when public)
- Review error logs in `log/`

**Scientific Questions:**
- Original THERAPI: inyoung.sung@snu.ac.kr
- Implementation: See documentation

---

**Last Updated:** November 2024

**Status:** ✅ Complete and Ready

**Start Here:** [FINAL_SUMMARY.md](FINAL_SUMMARY.md) or [QUICK_START.md](QUICK_START.md)
