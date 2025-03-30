import os
import shutil
import subprocess
from pathlib import Path
import json
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

# === CONFIGURATION ===
zero3_dir = "/data/scsgpu1/work/jackyjiang/saves/llama3.1-8b/full"
hf_dir = "/data/scsgpu1/work/jackyjiang/saves/llama3.1-8b-full-sft-limo-hf"
base_model_or_tokenizer = "meta-llama/Llama-3.1-8B-Instruct"  # or local tokenizer path
convert_to_safetensors = True
manual_tokenizer_model_path = ""  # optional fallback if auto tokenizer.model lookup fails

# === Ensure output dir exists ===
Path(hf_dir).mkdir(parents=True, exist_ok=True)

# === Step 1: Convert ZeRO-3 shards to full model weights ===
output_model_path = os.path.join(zero3_dir, "pytorch_model.bin")
print("🔄 Converting ZeRO-3 sharded checkpoint to full model...")
subprocess.run([
    "python", os.path.join(zero3_dir, "zero_to_fp32.py"),
    zero3_dir,
    output_model_path
], check=True)

# === Step 2: Convert to safetensors or keep .bin format ===
if convert_to_safetensors:
    if os.path.isdir(output_model_path):
        print("📁 Detected sharded checkpoint directory. Converting shards to .safetensors...")
        index_file = Path(output_model_path) / "pytorch_model.bin.index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"Missing sharded index file: {index_file}")
        
        with open(index_file) as f:
            index = json.load(f)

        # Handle both old and new Hugging Face index formats
        shard_filenames = sorted(set(
            v if isinstance(v, str) else v["filename"]
            for v in index["weight_map"].values()
        ))

        for fname in shard_filenames:
            shard_path = Path(output_model_path) / fname
            safetensors_path = Path(hf_dir) / fname.replace(".bin", ".safetensors")
            print(f"💾 Converting {shard_path.name} -> {safetensors_path.name}")
            state_dict = torch.load(shard_path, map_location="cpu")
            save_file(state_dict, safetensors_path)

        # Copy and patch index file for safetensors
        index_sft = Path(hf_dir) / "model.safetensors.index.json"
        shutil.copyfile(index_file, index_sft)
        with open(index_sft, "r+") as f:
            data = json.load(f)
            for k, v in data["weight_map"].items():
                if isinstance(v, str):
                    data["weight_map"][k] = v.replace(".bin", ".safetensors")
                elif isinstance(v, dict) and "filename" in v:
                    v["filename"] = v["filename"].replace(".bin", ".safetensors")
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
    else:
        print("💾 Converting single weight file to .safetensors...")
        state_dict = torch.load(output_model_path, map_location="cpu")
        safetensors_path = os.path.join(hf_dir, "model.safetensors")
        save_file(state_dict, safetensors_path)
else:
    print("📂 Keeping PyTorch weight format...")
    if os.path.isdir(output_model_path):
        for file in Path(output_model_path).glob("*"):
            shutil.copyfile(file, os.path.join(hf_dir, file.name))
    else:
        shutil.copyfile(output_model_path, os.path.join(hf_dir, "pytorch_model.bin"))

# === Step 3: Copy config/tokenizer JSON files ===
print("📋 Copying config and tokenizer JSON files...")
for fname in ["config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json"]:
    fpath = os.path.join(zero3_dir, fname)
    if os.path.exists(fpath):
        shutil.copyfile(fpath, os.path.join(hf_dir, fname))

# === Step 4: Extract or fallback to tokenizer.model ===
print("🔍 Trying to extract tokenizer.model from tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_or_tokenizer)
    tok_dir = tokenizer.name_or_path
    tok_model = os.path.join(tok_dir, "tokenizer.model")
    if os.path.exists(tok_model):
        shutil.copyfile(tok_model, os.path.join(hf_dir, "tokenizer.model"))
        print(f"✅ Found and copied tokenizer.model from: {tok_model}")
    else:
        raise FileNotFoundError(f"tokenizer.model not found at {tok_model}")
except Exception as e:
    print(f"⚠️ Could not auto-locate tokenizer.model: {e}")
    if manual_tokenizer_model_path and os.path.exists(manual_tokenizer_model_path):
        shutil.copyfile(manual_tokenizer_model_path, os.path.join(hf_dir, "tokenizer.model"))
        print(f"✅ Used fallback tokenizer.model: {manual_tokenizer_model_path}")
    else:
        print("🚨 tokenizer.model not found! Please copy it manually.")

print(f"\n🎉 Hugging Face-compatible model is ready at:\n{hf_dir}")
