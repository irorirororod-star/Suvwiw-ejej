import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import onnxruntime as ort
from huggingface_hub import hf_hub_download

MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX"

def load_model():
    print("🚀 Loading official ONNX model (optimized for CPU, maximum quality + speed)")
    model_path = hf_hub_download(MODEL_ID, subfolder="onnx", filename="model.onnx")
    
    # Download external data file if present
    try:
        hf_hub_download(MODEL_ID, subfolder="onnx", filename="model.onnx_data")
    except Exception:
        pass

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 0   # Use all CPU cores
    sess_options.inter_op_num_threads = 0
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )
    return session

def chunk_text(text: str, tokenizer, chunk_size: int = 512, overlap: int = 64):
    """Token-based chunking with overlap — perfect for long READMEs with mixed content."""
    # Simple markdown cleanup (optional, keeps code/headers intact)
    text = re.sub(r'\n{3,}', '\n\n', text)  # normalize excessive newlines
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)
    return chunks

def load_texts(input_path: str, tokenizer):
    path = Path(input_path)
    texts = []
    sources = []
    
    if path.suffix.lower() == '.jsonl':
        print("📄 JSONL mode: processing line-by-line")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        texts.append(data.get("text", line))
                    except json.JSONDecodeError:
                        texts.append(line)
                    sources.append(str(path))
    else:
        # Any text file (README.md, .txt, etc.) → full document + smart chunking
        print(f"📄 Text file mode: reading entire file + chunking ({path.name})")
        with open(path, encoding="utf-8") as f:
            full_text = f.read()
        
        chunks = chunk_text(full_text, tokenizer)
        texts.extend(chunks)
        sources.extend([str(path)] * len(chunks))
        print(f"   → Split into {len(chunks)} chunks")
    
    return texts, sources

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

        input_feed = {name: inputs[name] for name in [inp.name for inp in session.get_inputs()] 
                     if name in inputs}

        outputs = session.run(None, input_feed)
        output_names = [out.name for out in session.get_outputs()]
        emb_idx = next((i for i, name in enumerate(output_names) if "embedding" in name.lower()), 0)
        embedding = outputs[emb_idx]

        embeddings.append(embedding)

    return np.vstack(embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production ONNX EmbeddingGemma pipeline with smart chunking")
    parser.add_argument("--input", required=True, help="Input file (.jsonl, .md, .txt, etc.)")
    parser.add_argument("--output", default="embeddings.parquet")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prefix", default="task: search result | query: ",
                        help="Task prefix – strongly recommended")
    args = parser.parse_args()

    session = load_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    texts, sources = load_texts(args.input, tokenizer)
    print(f"✅ Loaded {len(texts):,} text chunks")

    embeddings = embed_texts(texts, session, tokenizer, args.batch_size, args.prefix)

    # Save with source tracking
    df = pd.DataFrame({
        "text": texts,
        "source_file": sources,
        "embedding": list(embeddings)
    })
    df.to_parquet(args.output, index=False, compression="zstd")
    
    size_mb = Path(args.output).stat().st_size / (1024**2)
    print(f"✅ Saved {len(texts):,} embeddings → {args.output} ({size_mb:.1f} MB)")
    print("   Ready for RAG / semantic search / vector DB")
