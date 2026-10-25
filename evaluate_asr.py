"""
Reference script for evaluating a fixed manifest with a Hugging Face Whisper model.
Expected manifest columns: audio_path, sentence. Audio should be readable by librosa.
Run in an environment with enough GPU/CPU memory.
"""
import argparse, time
import pandas as pd
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV with audio_path,sentence")
    parser.add_argument("--model", default="sajilck/whisper-small-malayalam-v2")
    parser.add_argument("--output", default="asr_results.csv")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    rows = []
    for _, row in df.iterrows():
        audio, sr = librosa.load(row["audio_path"], sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "input_features" in inputs and device == "cuda":
            inputs["input_features"] = inputs["input_features"].to(dtype)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=256)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        prediction = processor.batch_decode(generated, skip_special_tokens=True)[0]
        rows.append({
            "audio_path": row["audio_path"],
            "reference": row["sentence"],
            "prediction": prediction,
            "audio_duration_s": len(audio) / sr,
            "inference_time_s": elapsed,
            "rtf": elapsed / max(len(audio) / sr, 1e-9)
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")
    print("Mean inference time:", out["inference_time_s"].mean())
    print("Mean RTF:", out["rtf"].mean())

if __name__ == "__main__":
    main()
