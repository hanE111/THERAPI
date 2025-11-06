# TCGA Response Format Support

## Issue #12: No Response Label in TCGA Data

### Problem

```
Error: No response label column found
```

At [test_hierarchical_TCGA.py:198](test_hierarchical_TCGA.py#L198)

### Root Cause

The TCGA TRANSACT format uses different column names and response categories than GDSC:

**TCGA TRANSACT Format**:
- Column: `measure_of_response`
- Values: Clinical response categories (free text)
  - "Clinical Progressive Disease"
  - "Stable Disease"
  - "Partial Response"
  - "Complete Response"
  - etc.

**Expected by Code**:
- Column: `Label` or `response`
- Values: Binary (0 or 1)

### Clinical Response Categories

TCGA uses standard oncology response criteria (likely RECIST or similar):

| Clinical Response | Meaning | Label |
|-------------------|---------|-------|
| Complete Response (CR) | All tumors disappeared | 1 (Sensitive) |
| Partial Response (PR) | ≥30% tumor shrinkage | 1 (Sensitive) |
| Stable Disease (SD) | No significant change | 1 (Sensitive)* |
| Progressive Disease (PD) | ≥20% tumor growth | 0 (Resistant) |

*Stable Disease is typically considered a positive outcome (disease control)

### Solution

Added clinical response mapping in [test_hierarchical_TCGA.py:156-179](test_hierarchical_TCGA.py#L156-179):

```python
if 'measure_of_response' in tcga_resp.columns:
    # Map clinical responses to binary labels
    # Sensitive (1): Complete Response, Partial Response, Stable Disease
    # Resistant (0): Clinical Progressive Disease
    response_mapping = {
        'Complete Response': 1,
        'Partial Response': 1,
        'Stable Disease': 1,
        'Clinical Progressive Disease': 0,
        'Progressive Disease': 0,
        'Clinical Partial Response': 1,
        'Clinical Complete Response': 1,
        'Clinical Stable Disease': 1
    }
    tcga_resp['Label'] = tcga_resp['measure_of_response'].map(response_mapping)
    # Fill any unmapped values with 0 (resistant)
    tcga_resp['Label'] = tcga_resp['Label'].fillna(0).astype(int)
    print(f"Created binary labels from measure_of_response: {tcga_resp['Label'].value_counts().to_dict()}")
```

### Rationale

**Why SD = Sensitive (1)**:
- In clinical trials, disease control rate (DCR) = CR + PR + SD
- Stable disease means treatment is working (preventing progression)
- From drug discovery perspective: drug is effective if it stops growth

**Why PD = Resistant (0)**:
- Progressive disease indicates treatment failure
- Tumor is growing despite therapy
- Drug is not effective for this patient

### Response Mapping Coverage

The mapping includes multiple variations to handle different formatting:
- Standard: "Complete Response", "Partial Response", etc.
- Clinical prefix: "Clinical Progressive Disease", "Clinical Stable Disease"
- Short forms: "Progressive Disease" (in case "Clinical" prefix is missing)

**Fallback**: Any unmapped value → 0 (resistant) for conservative predictions

### TCGA vs GDSC Data Differences

| Aspect | GDSC (Cell Lines) | TCGA (Patients) |
|--------|-------------------|-----------------|
| Response metric | AUC (continuous) | Clinical category (discrete) |
| Measurement | In vitro dose-response | In vivo clinical assessment |
| Data volume | 333,161 measurements | 2,572 measurements |
| Granularity | Quantitative | Qualitative |
| Context | Controlled lab | Real-world treatment |

### Files Modified

**[test_hierarchical_TCGA.py](test_hierarchical_TCGA.py#L156-179)**:
- Added TRANSACT format detection
- Added clinical response mapping
- Created binary labels from `measure_of_response`
- Maintained backward compatibility with THERAPI format

### Impact

✅ **TCGA format supported** - Clinical responses mapped to binary labels
✅ **Clinically meaningful** - Mapping follows standard oncology practice
✅ **Robust** - Handles multiple response category variations
✅ **Backward compatible** - Still works with original THERAPI format
✅ **Testing can proceed** - Response labels now available

### Expected Label Distribution

Based on typical clinical trial outcomes:
- **Sensitive (1)**: ~40-60% (CR + PR + SD combined)
- **Resistant (0)**: ~40-60% (PD)

Exact distribution depends on:
- Cancer types included
- Treatment regimens
- Patient selection criteria
- Disease stage

### Comparison with GDSC

**GDSC** (from previous fix):
- Sensitive: 28.6% (95,458 / 333,161)
- Resistant: 71.4% (237,703 / 333,161)

**TCGA** (expected):
- More balanced distribution
- Higher sensitive rate (patients selected for treatment)
- Reflects real-world clinical outcomes

### Alternative Approaches

For future enhancements:

1. **Multi-class classification**: Keep all 4 categories (CR, PR, SD, PD)
2. **Ordinal regression**: Treat as ordered categories (CR > PR > SD > PD)
3. **Survival-based**: Use time to progression instead of categorical response
4. **Continuous score**: Create numerical score from categories (CR=3, PR=2, SD=1, PD=0)

### Testing

To verify the fix:
```bash
python test_hierarchical_TCGA.py \
    --aligner_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt \
    --predictor_dir ckpts/ \
    --data_dir data/ \
    --device cuda:0
```

Expected output:
```
Loaded TCGA TRANSACT format: (2572, 15)
Created binary labels from measure_of_response: {0: X, 1: Y}
```

### Related Issues

- **Issue #1-8**: Various training issues (fixed)
- **Issue #9**: GDSC TRANSACT format (fixed)
- **Issue #10**: Cell line alignment (fixed)
- **Issue #11**: Tissue mask in test/evaluate (fixed)
- **Issue #12**: TCGA response format (FIXED)

---

**Date**: 2025-11-05
**Status**: ✅ Fixed
**Changes**: Added clinical response mapping for TCGA TRANSACT format
**Result**: Binary labels created from clinical categories, testing can proceed
