# Configuration Structure Fix

## Issue #7: Config KeyError and Tensor Size Warning

### Problems

**1. KeyError: 'loss_weights'**
```python
KeyError: 'loss_weights'
```
At [train_hierarchical.py:110](train_hierarchical.py#L110)

**2. Tensor Broadcasting Warning**
```
UserWarning: Using a target size (torch.Size([])) that is different to the input size (torch.Size([1])).
This will likely lead to incorrect results due to broadcasting.
```
At [train_hierarchical.py:106](train_hierarchical.py#L106)

### Root Causes

#### 1. Nested Config Structure
The YAML config file uses a nested structure:
```yaml
training:
  loss_weights:
    center: 0.8
    routing: 0.4
    ...
```

But the code was trying to access it as a flat structure:
```python
config['loss_weights']  # ❌ KeyError!
```

#### 2. Scalar vs Tensor Dimension
The temperature parameter is a 1D tensor `[1]`, but the target was a scalar `[]`:
```python
torch.tensor(target_temp)  # Shape: []
model.tissue_router.temperature  # Shape: [1]
```

### Solutions

#### 1. Flatten Config on Load

Modified [train_hierarchical.py:300-337](train_hierarchical.py#L300-337) to flatten the nested YAML structure:

```python
if os.path.exists(args.config):
    with open(args.config, 'r') as f:
        config_raw = yaml.safe_load(f)

    # Flatten nested config for easier access
    config = {
        'batch_size': config_raw.get('training', {}).get('batch_size', 128),
        'latent_dim': config_raw.get('model', {}).get('latent_dim', 128),
        'learning_rate': config_raw.get('training', {}).get('learning_rate', 1e-3),
        'n_epochs': config_raw.get('training', {}).get('n_epochs', 200),
        'routing_strategy': config_raw.get('model', {}).get('routing_strategy', 'gumbel'),
        'target_temperature': config_raw.get('training', {}).get('target_temperature', 0.5),
        'loss_weights': config_raw.get('training', {}).get('loss_weights', {
            'center': 0.8,
            'routing': 0.4,
            'entropy': 0.2,
            'diversity': 0.1,
            'temperature': 0.1
        })
    }
```

This extracts values from the nested structure and creates a flat dictionary that matches the code's expectations.

#### 2. Fix Temperature Tensor Dimensions

Modified [train_hierarchical.py:106-107](train_hierarchical.py#L106-107):

```python
# Before:
temp_loss = F.mse_loss(model.tissue_router.temperature, torch.tensor(target_temp).to(...))

# After:
target_temp_tensor = torch.tensor([target_temp], dtype=model.tissue_router.temperature.dtype).to(...)
temp_loss = F.mse_loss(model.tissue_router.temperature, target_temp_tensor)
```

Now both tensors have shape `[1]` and proper dtype matching.

### Files Modified

**[train_hierarchical.py](train_hierarchical.py)**:
- Lines 300-337: Config flattening logic
- Lines 106-107: Temperature tensor dimension fix

### Impact

✅ **Config access works correctly** - All nested values properly extracted
✅ **No KeyError** - `loss_weights` accessible at top level
✅ **No dimension warnings** - Temperature tensors have matching shapes
✅ **Training can proceed** - Loss computation works without errors

### Config File Structure

The [hierarchical_config.yaml](configs/hierarchical_config.yaml) remains organized with nested structure for readability:

```yaml
model:
  latent_dim: 128
  routing_strategy: 'gumbel'
  ...

training:
  batch_size: 128
  learning_rate: 0.001
  n_epochs: 200
  loss_weights:
    center: 0.8
    routing: 0.4
    entropy: 0.2
    diversity: 0.1
    temperature: 0.1
```

The training script automatically flattens this structure for internal use.

### Related Issues

- **Issue #1**: Import conflict (fixed)
- **Issue #2**: Gzip pickle files (fixed)
- **Issue #3**: NumPy 2.x incompatibility (fixed)
- **Issue #4**: Zero common genes (fixed)
- **Issue #5**: CUDA compatibility (fixed)
- **Issue #6**: Tissue mask dimensions (fixed)
- **Issue #7**: Config structure & temperature tensor (FIXED)

---

**Date**: 2025-11-05
**Status**: ✅ Fixed
**Changes**: Config flattening + tensor dimension fix
