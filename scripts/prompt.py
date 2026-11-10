"""Type prompts to one of our trained Oromo models.

Models
  base      oromo-gpt-23m       the first pretrained model (Phase 3)
  base-tg   oromo-gpt-23m-tg    retrained with the Telegram data (default)
  instruct  oromo-gpt-23m-instruct   instruction-tuned (Phase 5)

The base models continue whatever text you start. The instruct model takes an
instruction; to give it an input text too, write: instruction ||| input text

Commands at the prompt: :t 0.8 (temperature, 0 = greedy), :k 40 (top-k),
:n 80 (max new tokens), :q (quit)

Usage: python scripts/prompt.py [base|base-tg|instruct]
Runs on the CPU, so it does not compete with training on the Mac GPU.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.generate import generate  # noqa: E402
from oromo_llm.instruct import respond  # noqa: E402
from oromo_llm.model import GPTModel  # noqa: E402
from oromo_llm.tokenizer import get_tokenizer  # noqa: E402

RUNS = {"base": "oromo-gpt-23m", "base-tg": "oromo-gpt-23m-tg", "instruct": "oromo-gpt-23m-instruct"}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "base-tg"
    if name not in RUNS:
        sys.exit(f"choose one of: {', '.join(RUNS)}")
    ckpt = torch.load(ROOT / "checkpoints" / RUNS[name] / "best.pt", map_location="cpu")
    model = GPTModel(ckpt["config"]).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    tok = get_tokenizer()
    ctx = ckpt["config"]["context_length"]
    temperature, top_k, max_new = 0.7, 50, 80

    print(f"\nModel: {RUNS[name]} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters)")
    if name == "instruct":
        print("Type an instruction. To add an input text: instruction ||| input text")
        print("Example: Oduun armaan gadii waa'ee maalii? Tokko filadhu: siyaasa, fayyaa, ispoortii, "
              "bashannanaa, teeknooloojii. ||| Atileetonni Oromoo maaraatoonii injifatan.")
    else:
        print("Type the beginning of a text and the model continues it. Example: Magaalaan Finfinnee")
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
        if name == "instruct":
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
