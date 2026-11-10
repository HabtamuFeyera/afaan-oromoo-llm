"""Type prompts to one of our trained Oromo models.

Models
  base      oromo-gpt-23m       the first pretrained model (Phase 3)
  base-tg   oromo-gpt-23m-tg    retrained with the Telegram data (default)
  instruct  oromo-gpt-23m-instruct   instruction-tuned (Phase 5)
  afrique   AfriqueQwen3.5-0.8B + our LoRA instruction tuning (Phase 6, the best model)

The base models continue whatever text you start. The instruct models take an
instruction; to give them an input text too, write: instruction ||| input text

Commands at the prompt: :t 0.8 (temperature, 0 = greedy), :k 40 (top-k),
:n 80 (max new tokens), :q (quit)

Usage: python scripts/prompt.py [base|base-tg|instruct|afrique]
The 23M models run on the CPU; "afrique" uses the Mac GPU and takes a few
seconds per answer.
"""

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.generate import generate  # noqa: E402
from oromo_llm.instruct import respond  # noqa: E402
from oromo_llm.model import GPTModel  # noqa: E402
from oromo_llm.tokenizer import get_tokenizer  # noqa: E402

RUNS = {"base": "oromo-gpt-23m", "base-tg": "oromo-gpt-23m-tg", "instruct": "oromo-gpt-23m-instruct",
        "afrique": "afrique-0.8b-instruct-lora"}
EXAMPLE = ("Oduun armaan gadii waa'ee maalii? Tokko filadhu: siyaasa, fayyaa, ispoortii, "
           "bashannanaa, teeknooloojii. ||| Atileetonni Oromoo maaraatoonii injifatan.")


class AfriqueModel:
    """AfriqueQwen3.5-0.8B with the LoRA adapter from scripts/18_instruct_open_model.py."""

    BASE = "McGill-NLP/AfriqueQwen3.5-0.8B-50Langs"

    def __init__(self):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        spec = importlib.util.spec_from_file_location("s18", ROOT / "scripts" / "18_instruct_open_model.py")
        self.s18 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.s18)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.BASE)
        base = AutoModelForCausalLM.from_pretrained(self.BASE, dtype=torch.float32)
        self.model = PeftModel.from_pretrained(base, ROOT / "checkpoints" / RUNS["afrique"] / "adapter")
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def answer(self, entry, max_new, temperature, top_k):
        prompt, _ = self.s18.encode(self.tok, entry, 384 - max_new, with_answer=False)
        x = torch.tensor([prompt], device=self.device)
        sample = temperature > 0
        out = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                                  do_sample=sample, temperature=temperature if sample else None,
                                  top_k=top_k if sample else None, eos_token_id=self.tok.eos_token_id,
                                  pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, len(prompt):], skip_special_tokens=True).strip()


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "base-tg"
    if name not in RUNS:
        sys.exit(f"choose one of: {', '.join(RUNS)}")

    if name == "afrique":
        print("Loading AfriqueQwen3.5-0.8B + LoRA adapter...")
        afrique = AfriqueModel()
        temperature, top_k, max_new = 0.0, 50, 60
        print(f"\nModel: AfriqueQwen3.5-0.8B + LoRA instruction tuning (on {afrique.device})")
        print("Type an instruction. To add an input text: instruction ||| input text")
        print(f"Example: {EXAMPLE}")
        print("Settings: :t <temperature>  :k <top-k>  :n <max tokens>  :q to quit\n")
    else:
        ckpt = torch.load(ROOT / "checkpoints" / RUNS[name] / "best.pt", map_location="cpu")
        model = GPTModel(ckpt["config"]).eval()
        model.load_state_dict(ckpt["model_state_dict"])
        tok = get_tokenizer()
        ctx = ckpt["config"]["context_length"]
        temperature, top_k, max_new = 0.7, 50, 80
        print(f"\nModel: {RUNS[name]} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters)")

    if name == "afrique":
        pass
    elif name == "instruct":
        print("Type an instruction. To add an input text: instruction ||| input text")
        print(f"Example: {EXAMPLE}")
    else:
        print("Type the beginning of a text and the model continues it. Example: Magaalaan Finfinnee")
    if name != "afrique":
        print("Settings: :t <temperature>  :k <top-k>  :n <max tokens>  :q to quit\n")

    while True:
        try:
            line = input(f"[t={temperature} k={top_k} n={max_new}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == ":q":
            break
        if line[:3] in (":t ", ":k ", ":n "):
            value = line[3:].strip()
            if line[1] == "t":
                temperature = float(value)
            elif line[1] == "k":
                top_k = int(value) or None
            else:
                max_new = int(value)
            continue

        torch.manual_seed(len(line))
        if name == "afrique":
            instruction, _, text = (part.strip() for part in line.partition("|||"))
            entry = {"task": "", "instruction": instruction, "input": text}
            print(f"\n{afrique.answer(entry, max_new, temperature, top_k)}\n")
        elif name == "instruct":
            instruction, _, text = (part.strip() for part in line.partition("|||"))
            entry = {"task": "", "instruction": instruction, "input": text}
            answer = respond(model, entry, "cpu", max_new_tokens=max_new,
                             temperature=temperature, top_k=top_k if temperature > 0 else None)
            print(f"\n{answer}\n")
        else:
            idx = torch.tensor([tok.encode(line)])
            out = generate(model, idx, max_new, ctx, temperature=temperature,
                           top_k=top_k if temperature > 0 else None, eos_id=tok.eot_token)
            print(f"\n{tok.decode(out[0].tolist())}\n")


if __name__ == "__main__":
    main()
