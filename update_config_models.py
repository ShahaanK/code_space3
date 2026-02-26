#!/usr/bin/env python3
"""
Update config2.yaml with all model entries needed by run_models.py.
Run this ONCE on Apophis after pulling the new code.

Usage:
    python update_config_models.py
"""

import yaml

CONFIG_FILE = "config2.yaml"

# All models that run_models.py may reference
ALL_MODELS = [
    # Local vLLM models — Llama (official HF quants)
    {"name": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
     "provider": "local_vllm", "temperature": 0, "max_tokens": 1024, "enabled": False},
    # Local vLLM models — Llama (casperhansen, legacy)
    {"name": "casperhansen/llama-3-70b-instruct-awq",
     "provider": "local_vllm", "temperature": 0, "max_tokens": 1024, "enabled": False},
    # Qwen AWQ
    {"name": "Qwen/Qwen2.5-72B-Instruct-AWQ",
     "provider": "local_vllm", "temperature": 0, "max_tokens": 1024, "enabled": False},
    # Qwen GPTQ (fallback)
    {"name": "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
     "provider": "local_vllm", "temperature": 0, "max_tokens": 1024, "enabled": False},
    # DeepSeek R1 Distill
    {"name": "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ",
     "provider": "local_vllm", "temperature": 0, "max_tokens": 1024, "enabled": False},
    # Mistral Small
    {"name": "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",
     "provider": "local_vllm", "temperature": 0, "max_tokens": 1024, "enabled": False},
    # AceGPT
    {"name": "FreedomIntelligence/AceGPT-v2-70B-chat",
     "provider": "local_vllm", "temperature": 0, "max_tokens": 1024, "enabled": False},
    # Cloud models (OpenRouter)
    {"name": "anthropic/claude-sonnet-4",
     "provider": "openrouter", "temperature": 0, "max_tokens": 300, "enabled": False},
    {"name": "openai/gpt-4o-mini",
     "provider": "openrouter", "temperature": 0, "max_tokens": 300, "enabled": False},
    {"name": "openai/gpt-4o",
     "provider": "openrouter", "temperature": 0, "max_tokens": 300, "enabled": False},
]


def main():
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    existing_names = {m["name"] for m in config.get("models", [])}
    added = []

    for model in ALL_MODELS:
        if model["name"] not in existing_names:
            config.setdefault("models", []).append(model)
            added.append(model["name"])

    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False,
                  width=120, allow_unicode=True)

    if added:
        print(f"Added {len(added)} new model entries to {CONFIG_FILE}:")
        for name in added:
            print(f"  + {name}")
    else:
        print(f"All model entries already present in {CONFIG_FILE}.")

    print(f"\nTotal models in config: {len(config['models'])}")
    for m in config["models"]:
        status = "ON " if m.get("enabled") else "OFF"
        print(f"  [{status}] {m['name']}")


if __name__ == "__main__":
    main()
