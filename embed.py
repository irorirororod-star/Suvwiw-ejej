import argparse
import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import onnxruntime as ort
from tqdm import tqdm
from transformers import AutoTokenizer, logging as tf_logging
from huggingface_hub import hf_hub_download

# --- Environment Hardening ---
# Suppress framework advisory warnings (PyTorch/TF not found)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# Disable tokenizer parallelism to avoid deadlocks in multi-threaded environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("ARCHON-EMBED")

# Suppress specific library noise
tf_logging.set_verbosity_error()
ort.set_default_logger_severity(3)  # Error level only

MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX"
MAX_SEQ_LENGTH = 2048

def load_model() -> ort.InferenceSession:
    """
    Initializes the ONNX inference session with optimized CPU execution providers.
    """
    logger.info(f"🚀 Downloading/Loading model: {MODEL_ID}")
    
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_ID, 
            subfolder="onnx", 
            filename="model.onnx",
            token=os.getenv("HF_TOKEN")
        )
        
        # Attempt to download external weights if they exist (required for some ONNX exports)
        try:
            hf_hub_download(
                repo_id=MODEL_ID, 
                subfolder="onnx", 
                filename="model.onnx_data",
                token=os.getenv("HF_TOKEN")
            )
        except Exception:
            # Not all models have external data files
            pass
            
    except Exception as e:
        logger.error(f"Failed to download model from HF Hub: {str(e)}")
        raise RuntimeError("Model acquisition failure.") from e

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 0  
    sess_options.inter_op_num_threads = 0
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL # More stable for CPU-bound batching

    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        return session
    except Exception as e:
        logger.error(f"Failed to initialize ONNX session: {str(e)}")
        raise RuntimeError("Inference engine initialization failure.") from e

def chunk_text(text: str, tokenizer: Any, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Token-based intelligent chunking with overflow protection.
    """
    if not text or not text.strip():
        return []

    # Normalize excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Defensive: Pre-split by character count if text is extremely large (>1M chars)
    # to avoid tokenizer memory issues, though Gemma tokenizer is generally robust.
    
    # Suppress the 'Token indices sequence length is longer than...' warning
    # because we are intentionally encoding the full text to slice it.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        tokens = tokenizer.encode(
            text, 
            add_special_tokens=False, 
            truncation=False,
            verbose=False
        )

    if not tokens:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size # Fallback to no overlap if overlap >= chunk_size

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        # Skip chunks that are too small unless it's the only chunk
        if len(chunk_tokens) < 10 and len(tokens) > 10:
            continue
            
        decoded_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
        if decoded_text:
            chunks.append(decoded_text)
            
    return chunks

def load_texts(input_path: str, tokenizer: Any) -> Tuple[List[str], List[str]]:
    """
    Loads and preprocesses text from supported file formats.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    texts: List[str] = []
    sources: List[str] = []
    
    if path.suffix.lower() == '.jsonl':
        logger.info("📄 Processing JSONL input")
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("text", "")
                    if content:
                        texts.append(content)
                        sources.append(f"{path}:{line_num}")
                except json.JSONDecodeError:
                    texts.append(line)
                    sources.append(f"{path}:{line_num}")
    else:
        logger.info(f"📄 Processing document: {path.name}")
        with open(path, encoding="utf-8") as f:
            full_text = f.read()
        
        chunks = chunk_text(full_text, tokenizer)
        texts.extend(chunks)
        sources.extend([str(path)] * len(chunks))
        logger.info(f"   → Generated {len(chunks)} chunks")
    
    return texts, sources

def embed_texts(
    texts: List[str], 
    session: ort.InferenceSession, 
    tokenizer: Any, 
    batch_size: int = 64, 
    prefix: str = ""
) -> np.ndarray:
    """
    Executes batch inference using the ONNX model.
    """
    if not texts:
        return np.array([])

    processed_texts = [prefix + t for t in texts] if prefix else texts
    all_embeddings: List[np.ndarray] = []
    
    # Map input names once
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    # Identify embedding output index
    emb_idx = 0
    for i, name in enumerate(output_names):
        if "embedding" in name.lower() or "last_hidden_state" in name.lower():
            emb_idx = i
            break

    for i in tqdm(range(0, len(processed_texts), batch_size), desc="Inference"):
        batch = processed_texts[i : i + batch_size]
        
        # Explicit truncation at model boundary to prevent indexing errors
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="np"
        )

        # Build feed dict dynamically based on model requirements
        input_feed = {name: inputs[name] for name in input_names if name in inputs}

        try:
            outputs = session.run(None, input_feed)
            batch_embeddings = outputs[emb_idx]
            
            # If model returns full hidden states, perform mean pooling
            if len(batch_embeddings.shape) == 3:
                # Simple mean pooling over sequence dimension
                mask = inputs.get("attention_mask")
                if mask is not None:
                    mask = np.expand_dims(mask, -1)
                    weighted_sum = np.sum(batch_embeddings * mask, axis=1)
                    sum_mask = np.sum(mask, axis=1)
                    batch_embeddings = weighted_sum / np.maximum(sum_mask, 1e-9)
                else:
                    batch_embeddings = np.mean(batch_embeddings, axis=1)

            all_embeddings.append(batch_embeddings)
        except Exception as e:
            logger.error(f"Inference error at batch {i//batch_size}: {str(e)}")
            raise

    return np.vstack(all_embeddings)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hardened ONNX EmbeddingGemma Pipeline")
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output", default="embeddings.parquet")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prefix", default="task: search result | query: ")
    args = parser.parse_args()

    try:
        # Load tokenizer first to validate environment
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.getenv("HF_TOKEN"))
        
        session = load_model()
        
        texts, sources = load_texts(args.input, tokenizer)
        if not texts:
            logger.warning("No text extracted from input. Exiting.")
            return

        logger.info(f"✅ Total chunks to embed: {len(texts):,}")
        embeddings = embed_texts(texts, session, tokenizer, args.batch_size, args.prefix)

        # Ensure directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            "text": texts,
            "source_file": sources,
            "embedding": list(embeddings)
        })
        
        df.to_parquet(args.output, index=False, compression="zstd")
        
        size_mb = output_path.stat().st_size / (1024**2)
        logger.info(f"✅ Success: {len(texts):,} embeddings -> {args.output} ({size_mb:.2f} MB)")
        
    except Exception as e:
        logger.error(f"Pipeline failure: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
