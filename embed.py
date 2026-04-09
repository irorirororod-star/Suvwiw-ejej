import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import onnxruntime as ort
from huggingface_hub import hf_hub_download

MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX"

def load_model(quantization: str = "q8"):
    if quantization == "q8":
        filename = "model_q8.onnx"
        print("🚀 Loading Q8_0 quantized model (nearly identical quality to fp32 + maximum CPU speed)")
    elif quantization == "q4":
        filename = "model_q4.onnx"
        print("⚡ Loading Q4_0 quantized model (max speed, minor quality trade-off)")
    else:  # fp32
        filename = "model.onnx"
        print("📈 Loading full fp32 model (maximum possible quality)")

    model_path = hf_hub_download(MODEL_ID, subfolder="onnx", filename=filename)
    
    # Download matching .onnx_data if it exists (required for most variants)
    try:
        data_filename = filename.replace(".onnx", ".onnx_data")
        hf_hub_download(MODEL_ID, subfolder="onnx", filename=data_filename)
    except Exception:
        pass  # Some small variants don't need separate data file

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 0   # Auto-use all CPU cores
    sess_options.inter_op_num_threads = 0
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )
    return session

def embed_texts(texts, session, tokenizer, batch_size=64, prefix=""):
    if prefix:
        texts = [prefix + t for t in texts]

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding with ONNX"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="np"
        )

        # Only feed inputs the model actually expects
        input_feed = {name: inputs[name] for name in [inp.name for inp in session.get_inputs()] 
                     if name in inputs}

        outputs = session.run(None, input_feed)

        # Get embedding output (robust to output naming)
        output_names = [out.name for out in session.get_outputs()]
        emb_idx = next((i for i, name in enumerate(output_names) if "embedding" in name.lower()), 0)
        embedding = outputs[emb_idx]

        embeddings.append(embedding)

    return np.vstack(embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production-grade ONNX EmbeddingGemma-300M pipeline (Q8_0 default)")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", default="embeddings.parquet")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quantization", choices=["q8", "q4", "fp32"], default="q8",
                        help="q8 = nearly identical quality + fastest practical speed")
    parser.add_argument("--prefix", default="task: search result | query: ",
                        help="Task prefix – strongly recommended for best quality")
    args = parser.parse_args()

    session = load_model(args.quantization)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load input texts
    texts = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    texts.append(data.get("text", line))
                except json.JSONDecodeError:
                    texts.append(line)

    print(f"✅ Loaded {len(texts):,} texts for embedding")

    embeddings = embed_texts(texts, session, tokenizer, args.batch_size, args.prefix)

    # Optional: Matryoshka truncation + L2 normalization (uncomment for smaller vectors)
    # embeddings = embeddings[:, :512]                    # e.g. 512d instead of 768d
    # embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Save as highly compressed Parquet
    df = pd.DataFrame({"text": texts, "embedding": list(embeddings)})
    df.to_parquet(args.output, index=False, compression="zstd")
    
    size_mb = Path(args.output).stat().st_size / (1024**2)
    print(f"✅ Saved {len(texts):,} embeddings → {args.output} ({size_mb:.1f} MB)")
    print(f"   Quantization used: {args.quantization.upper()}_0 | Quality: nearly identical to fp32")
