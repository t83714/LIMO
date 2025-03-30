import os
import shutil
import subprocess
from pathlib import Path
from transformers import AutoTokenizer

# === CONFIGURATION ===
zero3_dir = "/data/scsgpu1/work/jackyjiang/saves/llama3.1-8b/full"
hf_dir = "/data/scsgpu1/work/jackyjiang/saves/llama3.1-8b-full-sft-limo-hf"
base_model_or_tokenizer = "meta-llama/Llama-3.1-8B-Instruct"  # or path to the tokenizer you fine-tuned from
use_safetensors = False  # set to True if you want safetensors output instead

# Optional fallback if tokenizer.model is missing
manual_tokenizer_model_path = ""  # Leave blank if not needed

# === Ensure output dir exists ===
Path(hf_dir).mkdir(parents=True, exist_ok=True)

# === Step 1: Convert ZeRO-3 shards to full model weights ===
output_model_path = os.path.join(zero3_dir, "pytorch_model.bin")

print("🔄 Converting ZeRO-3 sharded checkpoint to full model...")
subprocess.run([
    "python", os.path.join(zero3_dir, "zero_to_fp32.py"),
    "--model_dir", zero3_dir,
    "--output_file", output_model_path
], check=True)

# === Step 2: Save model file to HF dir ===
if use_safetensors:
    import torch
    from safetensors.torch import save_file

    print("💾 Converting to .safetensors format...")
    state_dict = torch.load(output_model_path, map_location="cpu")
    safetensors_path = os.path.join(hf_dir, "model.safetensors")
    save_file(state_dict, safetensors_path)
else:
    shutil.copy(output_model_path, os.path.join(hf_dir, "pytorch_model.bin"))

# === Step 3: Copy config/tokenizer JSON files ===
print("📋 Copying config and tokenizer JSON files...")
for fname in ["config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json"]:
    fpath = os.path.join(zero3_dir, fname)
    if os.path.exists(fpath):
        shutil.copy(fpath, os.path.join(hf_dir, fname))

# === Step 4: Extract and copy tokenizer.model ===
print("🔍 Trying to extract tokenizer.model from tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_or_tokenizer)
    if hasattr(tokenizer, "sp_model"):
        tok_dir = tokenizer.name_or_path
        guess_path = os.path.join(tok_dir, "tokenizer.model")
        if os.path.exists(guess_path):
            shutil.copy(guess_path, os.path.join(hf_dir, "tokenizer.model"))
            print(f"✅ Found and copied tokenizer.model from: {guess_path}")
        else:
            raise FileNotFoundError
    else:
        raise ValueError("Not a SentencePiece-based tokenizer")

except Exception as e:
    print(f"⚠️ Could not auto-locate tokenizer.model from '{base_model_or_tokenizer}': {e}")
    if os.path.exists(manual_tokenizer_model_path):
        shutil.copy(manual_tokenizer_model_path, os.path.join(hf_dir, "tokenizer.model"))
        print(f"✅ Used fallback tokenizer.model: {manual_tokenizer_model_path}")
    else:
        print("🚨 tokenizer.model not found! Please check manually.")

print(f"\n🎉 Done! Hugging Face-compatible model saved at:\n{hf_dir}")
