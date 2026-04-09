import os
import sys

# --- LEVEL 0: PRE-EMPTIVE ENVIRONMENT HARDENING ---
# These must be set BEFORE any library imports to take effect during static initialization.
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ORT_LOGGING_LEVEL"] = "3"  # Suppress ONNX Runtime device discovery/PCI warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress potential TF fallback noise

import argparse
import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging immediately to capture all subsequent library initializations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ARCHON")

# Suppress verbose telemetry from third-party networking libraries
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import onnxruntime as ort
from tqdm import tqdm
from transformers import AutoTokenizer, logging as tf_logging
from huggingface_hub import hf_hub_download

# Silence transformers-specific logging
tf_logging.set_verbosity_error()

MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX"
MAX_SEQ_LENGTH = 2048

def get_hf_token() -> Optional[str]:
    """Retrieves the HF token from environment with validation."""
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not found. Proceeding with unauthenticated requests (Rate limits may apply).")
    return token

def load_model() -> ort.InferenceSession:
    """
    Initializes the ONNX inference session with optimized CPU execution providers.
    """
    token = get_hf_token()
    logger.info(f"🚀 Initializing Model: {MODEL_ID}")
    
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_ID, 
            subfolder="onnx", 
            filename="model.onnx",
            token=token
        )
        
        # Required external weights for Gemma ONNX architecture
        try:
            hf_hub_download(
                repo_id=MODEL_ID, 
                subfolder="onnx", 
                filename="model.onnx_data",
                token=token
            )
        except Exception:
            pass # Not all exports utilize external data files
            
    except Exception as e:
        logger.error(f"Critical failure during model acquisition: {str(e)}")
        sys.exit(1)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Optimization for GitHub Runners (typically 2-core)
    # Setting to 0 allows ORT to auto-detect, but we enforce sequential mode for stability.
    sess_options.intra_op_num_threads = 0  
    sess_options.inter_op_num_threads = 0
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL 

    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        return session
    except Exception as e:
        logger.error(f"Inference engine failed to initialize: {str(e)}")
        sys.exit(1)

def chunk_text(text: str, tokenizer: Any, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Token-based intelligent chunking with overflow protection and warning suppression.
    """
    if not text or not text.strip():
        return []

    # Normalize excessive whitespace to reduce token count
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Suppress the 'Token indices sequence length is longer than...' warning
    # during the document-level encoding pass.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        tokens = tokenizer.encode(
            text, 
            add_special_tokens=False, 
            truncation=False
        )

    if not tokens:
        return []

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        # Ensure we don't create empty or near-empty trailing chunks
        if len(chunk_tokens) < 5 and len(tokens) > 5:
            continue
            
        decoded_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
        if decoded_text:
            chunks.append(decoded_text)
            
    return chunks

def load_texts(input_path: str, tokenizer: Any) -> Tuple[List[str], List[str]]:
    """
    Loads and preprocesses text from supported file formats with robust error handling.
    """
    path = Path(input_path)
    if not path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    texts: List[str] = []
    sources: List[str] = []
    
    if path.suffix.lower() == '.jsonl':
        logger.info(f"📄 Processing JSONL: {path.name}")
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    content = data.get("text", line)
                    texts.append(content)
                    sources.append(f"{path.name}:{i}")
                except json.JSONDecodeError:
                    texts.append(line)
                    sources.append(f"{path.name}:{i}")
    else:
        logger.info(f"📄 Processing Document: {path.name}")
        with open(path, encoding="utf-8") as f:
            full_text = f.read()
        
        chunks = chunk_text(full_text, tokenizer)
        texts.extend(chunks)
        sources.extend([str(path)] * len(chunks))
        logger.info(f"   → Segmented into {len(chunks)} chunks")
    
    return texts, sources

def embed_texts(
    texts: List[str], 
    session: ort.InferenceSession, 
    tokenizer: Any, 
    batch_size: int = 64, 
    prefix: str = ""
) -> np.ndarray:
    """
    Executes batch inference using the ONNX model with explicit boundary enforcement.
    """
    if not texts:
        return np.array([])

    processed_texts = [prefix + t for t in texts] if prefix else texts
    all_embeddings: List[np.ndarray] = []
    
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    # Identify embedding output (usually index 0 or named 'last_hidden_state')
    emb_idx = 0
    for i, name in enumerate(output_names):
        if any(k in name.lower() for k in ["embedding", "last_hidden_state", "output"]):
            emb_idx = i
            break

    for i in tqdm(range(0, len(processed_texts), batch_size), desc="Inference"):
        batch = processed_texts[i : i + batch_size]
        
        # Hard boundary enforcement: truncation=True ensures no input exceeds MAX_SEQ_LENGTH
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="np"
        )

        input_feed = {name: inputs[name] for name in input_names if name in inputs}

        try:
            outputs = session.run(None, input_feed)
            batch_embeddings = outputs[emb_idx]
            
            # Handle models returning [batch, seq, hidden] vs [batch, hidden]
            if len(batch_embeddings.shape) == 3:
                # Apply mean pooling over the sequence dimension
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
            logger.error(f"Inference failure at batch {i//batch_size}: {str(e)}")
            raise

    return np.vstack(all_embeddings)

def main() -> None:
    parser = argparse.ArgumentParser(description="Systemically Hardened ONNX Embedding Pipeline")
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output", default="embeddings.parquet")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prefix", default="task: search result | query: ")
    args = parser.parse_args()

    try:
        token = get_hf_token()
        
        # Initialize tokenizer (SentencePiece requirement)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
        
        session = load_model()
        
        texts, sources = load_texts(args.input, tokenizer)
        if not texts:
            logger.warning("Extraction yielded zero content. Terminating.")
            return

        logger.info(f"✅ Total Chunks: {len(texts):,}")
        embeddings = embed_texts(texts, session, tokenizer, args.batch_size, args.prefix)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            "text": texts,
            "source_file": sources,
            "embedding": list(embeddings)
        })
        
        df.to_parquet(args.output, index=False, compression="zstd")
        
        size_mb = output_path.stat().st_size / (1024**2)
        logger.info(f"✅ Success: {len(texts):,} embeddings saved to {args.output} ({size_mb:.2f} MB)")
        
    except Exception as e:
        logger.error(f"Unhandled Pipeline Exception: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
