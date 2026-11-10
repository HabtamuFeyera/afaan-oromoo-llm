"""Text generation from chapters 4-5 of the book (listings 4.8 and 5.4)."""

import torch

from oromo_llm.tokenizer import get_tokenizer


def text_to_token_ids(text, tokenizer=None):
    tokenizer = tokenizer or get_tokenizer()
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # add batch dimension


def token_ids_to_text(token_ids, tokenizer=None):
    tokenizer = tokenizer or get_tokenizer()
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Book listing 4.8: greedy decoding."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(torch.softmax(logits, dim=-1), dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """Book listing 5.4: temperature scaling and top-k sampling.

    One small fix: the top-k threshold is kept 2-D so batches larger than 1 work.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, [-1]]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf"), device=logits.device), logits)

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def complete(model, prompt, max_new_tokens=60, temperature=0.8, top_k=40, device=None):
    """Convenience wrapper: prompt text in, completed text out."""
    tokenizer = get_tokenizer()
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    out = generate(model, idx, max_new_tokens, model.pos_emb.weight.shape[0],
                   temperature=temperature, top_k=top_k, eos_id=tokenizer.eot_token)
    if was_training:
        model.train()
    return token_ids_to_text(out.cpu(), tokenizer)
