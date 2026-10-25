# Malayalam Speech-in / Text-response Assistant — Vaani Research Internship

This repository contains the code, experiment notes, and report for a small reproducible Malayalam speech AI study. It evaluates a fine-tuned Whisper ASR model on a fixed Common Voice Malayalam test sample and performs a preliminary text-only QA pilot comparing direct answers with context-assisted answers.

## Status and scope (read before interpreting results)

This submission is **a partial experimental prototype**, not a completed 100-question audio QA evaluation. The ASR comparison used 100 fixed Common Voice test utterances. The QA comparison currently has only a 10-question text-only pilot. The full end-to-end audio → ASR → LLM response pipeline, 100-question QA test, two-condition acoustic QA evaluation, and formal manual hallucination annotation were not completed in the available run. Do not present the 10-question pilot as meeting the 100-example requirement.

## Reported results

### ASR (100 fixed Common Voice Malayalam test examples)

| Model | WER | CER | Mean inference time | Overall RTF |
|---|---:|---:|---:|---:|
| `sajilck/whisper-small-malayalam` | 58.20% | 12.67% | 2.180 s | 0.481 |
| `sajilck/whisper-small-malayalam-v2` | 23.40% | 3.75% | 2.208 s | 0.486 |

The v2 model reduced WER by approximately 59.8% relative and CER by approximately 70.4% relative on this fixed sample. Latency was similar; do not claim a speed improvement. Sample is scripted/read speech, not spontaneous conversation. These are results reported from the Colab run; the result CSVs are stored in the user's Google Drive and should be copied into `results/` before public release if permitted.

Beam search with `num_beams=5` did not change any of the 100 predictions and did not improve WER/CER in the observed run. Treat this as a negative result; generation configuration may have made the parameter ineffective, so it is not proof beam search never helps.

### QA (10-question text-only pilot)

- No-context exact-answer substring match: 0/10 (0%).
- Context-assisted exact-answer substring match: 6/10 (60%).
- Mean generation latency: baseline 2.833 s; context-assisted 2.131 s.
- These are simple string-match metrics on a small pilot, not semantic answer correctness. Manual review found some answers incomplete or malformed. Timing differences on 10 examples are noisy and not evidence that context makes generation faster.

## Dataset

Mozilla Common Voice Scripted Speech 27.0 Malayalam, official dataset page:
https://mozilladatacollective.com/datasets/cmu5sv0160040o107vt0r9nt6

The inspected test split contained 894 rows; all 894 referenced audio files were present. A reproducible 100-example manifest was selected using random seed 42. Dataset license is reported as CC0-1.0, but follow dataset terms and speaker privacy rules. Do not upload raw audio or the full dataset archive to GitHub. The repository should contain scripts, metrics, and instructions only.

## Reproduction

The original work was run in Google Colab with a Tesla T4 GPU, PyTorch 2.11.0+cu130, Transformers 5.20.1. Paths in the original notebook point to Google Drive; change them for your environment.

1. Obtain the Common Voice Malayalam dataset through the official dataset page and follow its terms.
2. Place the extracted data so `DATA_DIR` points to the folder containing `test.tsv` and `clips/`.
3. Run `notebooks/01_dataset_inspection.ipynb` after copying the notebook from your working Colab session (the uploaded notebook may contain absolute Google Drive paths).
4. Use `src/evaluate_asr.py` as a reference implementation for fixed-manifest ASR evaluation.
5. For QA pilot metrics, use the `qa_seed_comparison.csv` from the experiment folder and run `src/evaluate_qa_pilot.py`.

## Contents

- `report/final_report.pdf`: report with architecture comparison, observed experiments, limitations, and references.
- `report/task1_architecture_comparison.md`: standalone answer to Task 1.
- `src/`: reproducibility scripts.
- `results/`: summary metrics only; add actual result CSVs from your Drive if allowed.
- `notebooks/`: place your working Colab notebook here.
- `requirements.txt`: approximate Python dependencies.

## Ethical and reporting notes

- Do not claim unrun experiments or fabricated results.
- The QA pilot is not a 100-question evaluation.
- Do not include credentials, access tokens, or dataset audio in this repository.
- Report that Common Voice is scripted/read speech and limits generalization to real-time conversation.
