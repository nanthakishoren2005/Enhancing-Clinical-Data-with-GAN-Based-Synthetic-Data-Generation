# Task 5 Filtered Re-Test — Investigation Report

## 1. Corrected Task 5 Answer (which number is correct)

**Report 2 (91.75% accuracy / 91.77% macro F1) is correct.** Report 1 (94.50%) is wrong.

The clean re-evaluation (`outputs/v2/filtered_reeval.csv`) confirms both numbers were evaluated on the same untouched 400-image seed-42 test set:

| Model | Accuracy | Macro F1 | Note |
|-------|----------|----------|------|
| Baseline (verified reference) | 92.75% | 92.77% | Existing `checkpoints/baseline_best.pt` |
| Filtered-augmented (clean run) | **91.75%** | **91.77%** | `outputs/v2/filtered_augmented_full_best.pt` |

Conclusion: filtered augmentation on 187 kept synthetics performs **slightly worse** than the verified baseline (−1.00 percentage point on accuracy), contradicting the earlier erroneous claim of +1.75%.

Kept images per class (threshold 0.90, capped at 100/class): normal=0, covid=100, lung_opacity=2, viral_pneumonia=85, total=187.

## 2. Exact Cause of the Contradiction

**The first run (94.50%) was trained on a contaminated `filtered_synthetic/` directory containing exactly 2× duplicate images per class.**

Evidence from file timestamps:
- Smoke test (earlier) copied images into `outputs/v2/smoke_filter_out`, then `run_filtered_experiment.py` Step 1 ran `filter_synthetic` into `outputs/v2/filtered_synthetic/` **without clearing it first**.
- `filter_synthetic` does not overwrite existing same-named files (it adds a `_k` suffix), so duplicate copies accumulated.
- Per-class kept counts **before** capping:
  - First (contaminated) run: normal=0, covid=150→shown as 300 available, lung_opacity=2→shown as 4 available, viral_pneumonia=85→shown as 170 available (exactly 2× for covid, lung_opacity, viral_pneumonia)
  - Second (clean) run after removing smoke artifacts: normal=0, covid=150, lung_opacity=2, viral_pneumonia=85 (total 237 kept, 187 after capping)
- After capping at 100/class:
  - Contaminated run: 204 training images → reported 94.50%
  - Clean run: 187 training images → reported 91.75%

**The checkpoint and CSV were overwritten** between the two reports:
- Both runs wrote to `outputs/v2/filtered_augmented_full_best.pt` and `outputs/v2/filtered_comparison.csv`.
- The first run's checkpoint and CSV row were replaced by the second run's outputs (LastWriteTime 21:02:49 for the checkpoint).
- Only the clean-run data (91.75%) survived on disk; the 94.50% number existed only in the transient output of the first run.

**Both evaluations ran on the same untouched 400-image seed-42 test set.** No split difference was involved — the gap came entirely from the contaminated training set.

## 3. Repeated-Split Evaluation (Full Run, seeds 42/43/44)

Results (`outputs/v2/repeated_eval_full.csv`):

| Seed | Baseline Acc | Baseline Macro F1 | Augmented Acc | Augmented Macro F1 |
|------|-------------|-------------------|---------------|-------------------|
| 42 | 90.75% | 90.66% | 91.75% | 91.77% |
| 43 | 89.75% | 89.76% | 91.75% | 91.75% |
| 44 | 92.50% | 92.54% | 92.00% | 92.00% |

**Mean ± std across 3 seeds:**

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Baseline | **91.00% ± 1.14%** | **90.99% ± 1.16%** |
| Filtered-Augmented | **91.83% ± 0.12%** | **91.84% ± 0.11%** |

**Does filtered augmentation beat the baseline?** On average yes (+0.83 percentage points accuracy), but the margin is small and highly run-dependent:
- Seed 42: augmented wins by +1.00 pp
- Seed 43: augmented wins by +2.00 pp
- Seed 44: baseline wins by −0.50 pp

The augmented model's extremely low std (±0.12%) vs baseline's higher std (±1.14%) suggests filtered augmentation makes results more stable across splits, but the absolute gain is modest and not statistically decisive with only 3 seeds. Filtered augmentation does **not** reliably beat the verified baseline of 92.75% — two of three seeds are below it.

## 4. Reporting Guidance

**What CAN be reported in project documents:**
- The verified baseline (92.75% / 92.77%) remains the reference number — unchanged.
- Task 5 filtered re-test result: 91.75% accuracy / 91.77% macro F1 on 187 kept filtered synthetics (clean run), evaluated on the untouched seed-42 test set.
- Repeated-split means: baseline 91.00% ± 1.14%, filtered-augmented 91.83% ± 0.12% across seeds 42/43/44.
- The kept-image breakdown (normal=0, covid=100, lung_opacity=2, viral_pneumonia=85; 39.5% keep rate at threshold 0.90).
- That the low-data arm is blocked pending `data_low/` (Kaggle results zip).

**What must NOT be reported yet:**
- **Do NOT report 94.50%** — it was produced by a contaminated training set (2× duplicate synthetics) and is invalid.
- **Do NOT claim filtered augmentation reliably beats baseline** — only 2 of 3 seeds show a gain, and all are below the verified 92.75% baseline.
- **Do NOT report any low-data (Experiment 3) filtered result** — `data_low/` is not present on this machine.
- **Do NOT overwrite** `filtered_augmented_full_best.pt` or `filtered_comparison.csv` — they are now read-only reference artifacts.

## New Files Created (all under outputs/v2/ or new scripts)

- `scripts/evaluate_reeval.py` — clean re-evaluation script
- `outputs/v2/filtered_reeval.csv` — Task 2 clean re-eval results
- `outputs/v2/repeated_eval_full.csv` — Task 3 full repeated-split results
- `outputs/v2/filtered_investigation.md` — this report

No existing files were modified, overwritten, or deleted.