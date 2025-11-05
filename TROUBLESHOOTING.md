# Hierarchical THERAPI - Troubleshooting Guide

## Common Issues and Solutions

### 1. ModuleNotFoundError: No module named 'utils.data_loader'

**Symptom:**
```
ModuleNotFoundError: No module named 'utils.data_loader'; 'utils' is not a package
```

**Cause:**
Naming conflict between `src/utils.py` (file) and `utils/` (package).

**Solution:**
✅ **FIXED** - The implementation now properly handles this with:
- `utils/__init__.py` created
- `models/__init__.py` created
- All import statements updated

See [IMPORT_FIX.md](IMPORT_FIX.md) for details.

**Verification:**
```bash
python -c "from utils.data_loader import TransactDataLoader; print('✓ Works')"
```

---

### 2. Import Error from src/

**Symptom:**
```
ImportError: cannot import name 'set_seed' from 'utils'
```

**Cause:**
Conflicting import paths.

**Solution:**
The scripts now use explicit path handling:
```python
# Import from src
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)
import utils as src_utils
sys.path.remove(src_path)

# Use aliased utilities
set_seed = src_utils.set_seed
```

---

### 3. CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solution:**

**Option A: Reduce batch size**
```yaml
# In configs/hierarchical_config.yaml
training:
  batch_size: 64  # Or 32
```

**Option B: Use CPU**
```bash
python train_hierarchical.py --device cpu
```

**Option C: Clear cache**
```python
import torch
torch.cuda.empty_cache()
```

---

### 4. Poor Routing Accuracy

**Symptom:**
Routing accuracy < 50%

**Causes & Solutions:**

**A. Insufficient training**
```yaml
training:
  n_epochs: 300  # Increase from 200
```

**B. Wrong loss weights**
```yaml
training:
  loss_weights:
    routing: 0.8  # Increase
    center: 1.0   # Increase
```

**C. Temperature issues**
```yaml
training:
  target_temperature: 0.3  # Lower (more decisive)
```

**D. Insufficient tissue coverage**
Check tissue distribution:
```python
from utils.tissue_mapping import TissueMapper
mapper = TissueMapper()
stats = mapper.get_tissue_statistics(cell_line_tissues)
print(stats)
# Ensure each tissue has >= 5 samples
```

---

### 5. Data Loading Errors

**Symptom:**
```
FileNotFoundError: data/GDSC/rnaseq/GDSC_rnaseq_data.pkl
```

**Solution:**

**Check data structure:**
```bash
ls -la data/GDSC/rnaseq/
ls -la data/TCGA/rnaseq/
```

**Expected structure:**
```
data/
├── GDSC/
│   └── rnaseq/
│       └── GDSC_rnaseq_data.pkl
├── TCGA/
│   └── rnaseq/
│       └── TCGA_rnaseq_data.pkl
...
```

**Verify file format:**
```python
import pickle
with open('data/GDSC/rnaseq/GDSC_rnaseq_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"Type: {type(data)}")
print(f"Shape: {data.shape if hasattr(data, 'shape') else len(data)}")
```

---

### 6. Model Not Converging

**Symptom:**
Loss not decreasing, staying high

**Causes & Solutions:**

**A. Learning rate too high**
```yaml
training:
  learning_rate: 0.0001  # Decrease
```

**B. Data not normalized**
```python
# In data_loader.py or your script
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(data)
```

**C. Exploding gradients**
```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**D. Wrong tissue mapping**
```python
# Verify tissue indices are valid
assert all(0 <= idx < n_tissues for idx in tissue_labels)
```

---

### 7. Slow Training

**Symptom:**
Training taking too long

**Solutions:**

**A. Use GPU**
```bash
python train_hierarchical.py --device cuda:0
```

**B. Reduce epochs for testing**
```yaml
training:
  n_epochs: 50  # Quick test
```

**C. Use smaller gene set**
```python
# Select top 500 most variable genes
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.1)
data_reduced = selector.fit_transform(data)
```

**D. Enable mixed precision**
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

---

### 8. Configuration Not Loading

**Symptom:**
```
FileNotFoundError: configs/hierarchical_config.yaml
```

**Solution:**

**Create default config:**
```bash
mkdir -p configs
cp configs/hierarchical_config.yaml.example configs/hierarchical_config.yaml
```

**Or specify path:**
```bash
python train_hierarchical.py --config /path/to/config.yaml
```

**Or run without config (uses defaults):**
```bash
python train_hierarchical.py  # Will use default values
```

---

### 9. Checkpoint Loading Error

**Symptom:**
```
RuntimeError: Error(s) in loading state_dict
```

**Causes & Solutions:**

**A. Wrong model architecture**
```python
# Make sure config matches training
checkpoint = torch.load('ckpts/model.pt')
config = checkpoint.get('config', {})
# Use same config for loading
```

**B. Partial loading**
```python
# Load with strict=False if model changed
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

**C. Device mismatch**
```python
# Always specify map_location
checkpoint = torch.load('model.pt', map_location='cpu')
# Then move to desired device
model.to(device)
```

---

### 10. Tissue Routing All Uniform

**Symptom:**
All samples route to same tissue or uniform distribution

**Causes & Solutions:**

**A. Entropy regularization too strong**
```yaml
training:
  loss_weights:
    entropy: 0.05  # Decrease from 0.2
```

**B. Temperature too high**
```yaml
training:
  target_temperature: 0.3  # Decrease
```

**C. Router not learning**
```python
# Check router gradients
for name, param in model.tissue_router.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
```

**D. Routing loss too weak**
```yaml
training:
  loss_weights:
    routing: 1.0  # Increase from 0.4
```

---

## Debugging Tips

### Check Model Output

```python
model.eval()
with torch.no_grad():
    output = model(patient_expr, cell_line_exprs)
    print("Tissue weights shape:", output['tissue_weights'].shape)
    print("Tissue weights sum:", output['tissue_weights'].sum(dim=1))
    print("Max weight:", output['tissue_weights'].max())
    print("Entropy:", -torch.sum(output['tissue_weights'] * torch.log(output['tissue_weights'] + 1e-8), dim=1))
```

### Monitor Training

```python
# In training loop
if epoch % 10 == 0:
    print(f"Epoch {epoch}:")
    print(f"  Temperature: {model.tissue_router.temperature.item()}")
    print(f"  Avg max weight: {tissue_weights.max(dim=1)[0].mean()}")
    print(f"  Avg entropy: {entropy.mean()}")
```

### Validate Data

```python
# Check for NaN/Inf
assert not torch.isnan(patient_expr).any()
assert not torch.isinf(patient_expr).any()

# Check value ranges
print(f"Patient expr range: [{patient_expr.min()}, {patient_expr.max()}]")
print(f"Cell line expr range: [{cell_line_exprs.min()}, {cell_line_exprs.max()}]")
```

### Test Small Batch

```python
# Test with small data first
small_patient = patient_expr[:2]  # 2 samples
small_cells = cell_line_exprs[:10]  # 10 cell lines

output = model(small_patient, small_cells)
print("Small batch works:", output['prediction'].shape)
```

---

## Getting Help

### Before Asking

1. ✅ Check this troubleshooting guide
2. ✅ Review [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)
3. ✅ Check [IMPORT_FIX.md](IMPORT_FIX.md) for import issues
4. ✅ Look at error logs in `log/`
5. ✅ Try with minimal example

### When Asking

Include:
- Error message (full traceback)
- Command you ran
- Python version: `python --version`
- PyTorch version: `python -c "import torch; print(torch.__version__)"`
- Config file (if using)
- Log file snippet
- What you've tried

### Resources

- **Documentation**: [INDEX.md](INDEX.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Workflow**: [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)
- **Original THERAPI**: inyoung.sung@snu.ac.kr

---

## Quick Fixes Checklist

Before diving deep, try these:

```bash
# 1. Verify Python environment
python --version  # Should be 3.9+
pip list | grep torch

# 2. Verify imports work
python -c "from utils.data_loader import TransactDataLoader; print('OK')"

# 3. Check data exists
ls data/GDSC/rnaseq/
ls data/TCGA/rnaseq/

# 4. Try minimal config
python train_hierarchical.py --help

# 5. Check logs
tail -50 log/*.log

# 6. Clean and retry
rm -rf __pycache__ utils/__pycache__ models/__pycache__
./run_hierarchical_pipeline.sh
```

---

**Last Updated**: November 2024

**Status**: Actively maintained

**For urgent issues**: Check [IMPORT_FIX.md](IMPORT_FIX.md) first
