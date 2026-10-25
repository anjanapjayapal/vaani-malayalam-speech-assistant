# Malayalam Speech AI Research Internship — Experimental Report
**Prepared:** 25 October 2026  
**Scope:** Malayalam ASR comparison and a preliminary context-assisted QA pilot

## Abstract
This project investigates a reproducible Malayalam speech-to-text component and a small text-response experiment as a first step toward an Indic speech assistant. On a fixed sample of 100 utterances from the Common Voice Malayalam test split, `sajilck/whisper-small-malayalam-v2` achieved WER 23.40% and CER 3.75%, compared with 58.20% WER and 12.67% CER for `sajilck/whisper-small-malayalam`. A separate 10-question text-only pilot compared Qwen2.5-1.5B-Instruct without context and with a supplied verified fact. Exact-answer substring match was 0% and 60%, respectively. The pilot suggests context can help for some simple factual questions, but it is too small and the metric too crude to establish semantic correctness. The 100-question QA evaluation and full end-to-end audio QA pipeline remain incomplete and are stated as limitations.

## 1. Objectives
1. Compare two recent speech-capable model architectures and discuss low-latency conversational limitations.
2. Evaluate Malayalam ASR reproducibly on a fixed test sample.
3. Explore whether supplying verified context improves small-model Malayalam factual QA.
4. Document limitations and a practical improvement plan.

## 2. Dataset and experimental setup
The ASR dataset was Mozilla Common Voice Scripted Speech 27.0 Malayalam. The inspected `test.tsv` had 894 rows and 13 columns; all 894 referenced clips were found. A fixed 100-example sample was selected with random seed 42 and reused across ASR model comparisons. The dataset is read/scripted speech, not spontaneous conversational speech. Dataset page: https://mozilladatacollective.com/datasets/cmu5sv0160040o107vt0r9nt6

The reported environment was Google Colab with a Tesla T4 GPU, PyTorch 2.11.0+cu130, Transformers 5.20.1. ASR used Hugging Face Transformers with float16 on CUDA. Timing can vary due to warm-up, caching, and runtime conditions.

## 3. Task 1: architecture comparison
See `task1_architecture_comparison.md` for the full comparison of Moshi and Qwen2.5-Omni, including representations, alignment, streaming, autoregressive generation, stopping, limitations, and proposed improvement.

## 4. ASR experiment
### 4.1 Results
| Model | N | WER | CER | Mean inference time | Overall RTF |
|---|---:|---:|---:|---:|---:|
| `sajilck/whisper-small-malayalam` | 100 | 58.20% | 12.67% | 2.180 s | 0.481 |
| `sajilck/whisper-small-malayalam-v2` | 100 | 23.40% | 3.75% | 2.208 s | 0.486 |

Relative WER reduction = (58.20 − 23.40) / 58.20 ≈ 59.8%. Relative CER reduction = (12.67 − 3.75) / 12.67 ≈ 70.4%. These results show a large accuracy improvement on the fixed sample but no demonstrated speed improvement; average latency and RTF are similar.

### 4.2 Negative experiment: beam search
A `num_beams=5` attempt produced the same 100 transcripts as baseline in the observed run, with WER 58.2% and CER 12.672%. The observed timing difference was small and likely not meaningful. Because generation configuration may not have applied as intended, this should be reported as a failed/ineffective experiment in this run, not proof that beam search cannot help.

### 4.3 Error analysis
Observed errors included word-boundary and spacing differences, spelling variants, proper names, and acronym confusion. Malayalam orthographic segmentation can make word-level WER sensitive to tokenization/spacing; CER is also reported to provide a character-level view. For final comparisons, normalization rules should be fixed and disclosed, and raw predictions retained for audit.

## 5. QA pilot: no context vs verified context
A 10-question text-only pilot used Qwen2.5-1.5B-Instruct with deterministic decoding (`do_sample=False`, `max_new_tokens=80`). The same questions were run without context and with a short verified fact supplied in the prompt.

| Condition | N | Exact-answer substring match | Mean latency |
|---|---:|---:|---:|
| No context | 10 | 0% | 2.833 s |
| Verified context | 10 | 60% | 2.131 s |

This substring metric is only a screening measure. It misses valid paraphrases (e.g., a Malayalam number word instead of a digit) and may mark a response correct even if the surrounding sentence is wrong. Manual inspection found incomplete or malformed answers even in the context condition. Thus, the result supports only the cautious statement that supplied context improved this pilot's exact-string match rate. It does not establish a general hallucination reduction or semantic accuracy improvement. Latency is measured for text generation only, not end-to-end speech response.

## 6. Improvement proposal
The most feasible next step is a retrieval-augmented pipeline: (1) transcribe audio with Whisper v2; (2) retrieve a trusted fact from a small curated knowledge set; (3) prompt the LLM to answer only from that context and abstain when evidence is missing; (4) record ASR confidence and response latency; and (5) manually annotate answer correctness, unsupported claims, and abstentions. A production system should also handle noisy speech, code-switching, endpointing, timeouts, and clarification requests.

## 7. Limitations and incomplete requirements
- Common Voice is scripted/read speech and cannot establish performance on spontaneous dialogue.
- The ASR results cover 100 fixed examples, but QA currently covers only 10 text-only questions.
- The full audio → ASR → LLM end-to-end evaluation was not completed in this run.
- Two acoustic/linguistic QA conditions and formal hallucination-rate annotation remain outstanding.
- The 10-question substring metric is not a substitute for human-validated semantic correctness.
- No fine-tuning was performed. Results are exploratory and should not be generalized beyond the tested sample.

## 8. Conclusion
The strongest completed result is the ASR comparison: the v2 Malayalam Whisper checkpoint substantially reduced WER and CER on the fixed sample, with similar latency. The QA pilot showed that adding verified context improved exact-answer matching, but the small sample and crude metric prevent strong claims. The proposed next step is a fully reproducible 100-question retrieval-assisted evaluation connected to audio input.

## References
1. Défossez, A. et al. (2024). *Moshi: a speech-text foundation model for real-time dialogue*. arXiv:2410.00037. https://arxiv.org/abs/2410.00037
2. Xu, J. et al. (2025). *Qwen2.5-Omni Technical Report*. arXiv:2503.20215. https://arxiv.org/abs/2503.20215
3. Qwen Team (2025). *Qwen2.5-Omni: See, Hear, Talk, Write, Do It All!* https://qwenlm.github.io/blog/qwen2.5-omni/
4. Mozilla Common Voice Dataset, Malayalam, release 27.0. https://mozilladatacollective.com/datasets/cmu5sv0160040o107vt0r9nt6
5. Hugging Face model card: `sajilck/whisper-small-malayalam`. https://huggingface.co/sajilck/whisper-small-malayalam
6. Hugging Face model card: `sajilck/whisper-small-malayalam-v2`. https://huggingface.co/sajilck/whisper-small-malayalam-v2
7. Hugging Face model card: `Qwen/Qwen2.5-1.5B-Instruct`. https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
