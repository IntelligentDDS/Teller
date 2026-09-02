# TELLER

This repository contains the implementation for **TELLER: Non-Intrusive
Cross-Layer Root-Cause Analysis for LLM Inference**. It includes the
non-intrusive NVTX/CUPTI collector, Trace Pair Encoding (TPE), and multimodal
root-cause models used in the paper. The accompanying dataset is in the
sibling `../data` directory.（figshare:doi:10.6084/m9.figshare.33142493.v1）

## Installation

TELLER requires Python 3.10 or newer.

```bash
pip install -e .
pip install -e '.[mot,rootcause]'
```

For trace collection, install a CUDA toolkit with CUPTI and make `cmake`
available. Installing from source builds the CUPTI injection library when CUDA
is available.

## Trace Collection

Run an inference program under TELLER:

```bash
export TELLER_TRACE_DIR=/path/to/traces
teller run -- python inference.py
```

The collector writes per-process trace events and captured logs under
`TELLER_TRACE_DIR`. The `--cuda-home`, `--nvtx-json`, and `--so` options select
the CUDA installation, NVTX configuration, or prebuilt CUPTI library.

## Data Preparation and RCA Models

The released dataset contains aligned `trace.json`, `log.txt`, and
`annotation.json` files. Train a TPE vocabulary and tokenize the data with:

```bash
python scripts/train_tpe.py --data-dir ../data --output-dir ../artifacts/tpe
python scripts/tokenize_datasets.py \
  --data-dir ../data \
  --tokenizer-dir ../artifacts/tpe/<run-id> \
  --output-dir ../artifacts/tokenized
```

Train the dual-head RCA model after tokenization:

```bash
python scripts/train_mot.py \
  --config configs/mot/train.yaml \
  --data-dir ../artifacts/tokenized/<run-id> \
  --tpe-dir ../artifacts/tpe/<run-id>
```

The single-stream alternative is `scripts/train_rc_single.py`. Its
configuration specifies the backbone, deterministic split seed, and training
hyperparameters. Hugging Face model weights are downloaded on demand and are
not part of this archive.

## Repository Layout

| Path | Contents |
| --- | --- |
| `src/teller/` | Trace collection, parsing, TPE, and RCA models |
| `src/csrc/` | CUPTI injection library source |
| `scripts/` | TPE, result-summary, and model-training entry points |
| `configs/` | Training and tokenization configurations |
| `output/` | The paper's main results |
