"""Phase 4, step 2: news-topic classification on MasakhaNEWS Oromo (book chapter 6).

Setups
- majority:              always predicts the most common training topic
- tfidf_logreg:          TF-IDF word 1-2 grams + logistic regression (C tuned on dev)
- random_init_all:       our GPT architecture with random weights, all layers trained
- pretrained_last_block: our pretrained GPT, the book's recipe (last block + final norm + head)
- pretrained_all:        our pretrained GPT, all layers trained

The learning rate for each neural setup is picked on dev (seed 1), then the
best rate is rerun with more seeds. Test scores are reported as mean and
standard deviation over seeds, on the full test set and on the test set without
articles the pretraining text already contained (see 09_prepare_news.py).

Outputs: checkpoints/news-classifier/results.json and best_classifier.pt
Usage: python scripts/10_finetune_news.py [--quick]
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.classify import build_classifier, make_loader, predict, scores, train_classifier  # noqa: E402
from oromo_llm.train import get_device  # noqa: E402

NEWS = ROOT / "data" / "news"
OUT = ROOT / "checkpoints" / "news-classifier"
PRETRAINED = ROOT / "checkpoints" / "oromo-gpt-23m" / "best.pt"


def read_jsonl(split):
    return [json.loads(l) for l in (NEWS / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def score_test(preds, test_rows, num_classes):
    gold = [r["label_id"] for r in test_rows]
    clean = [i for i, r in enumerate(test_rows) if r["pretrain_overlap"] < 0.5]
    full = scores(preds, gold, num_classes)
    cl = scores([preds[i] for i in clean], [gold[i] for i in clean], num_classes)
    return {"accuracy": full["accuracy"], "macro_f1": full["macro_f1"],
            "per_class_f1": full["per_class_f1"],
            "clean_accuracy": cl["accuracy"], "clean_macro_f1": cl["macro_f1"]}


def mean_std(runs, key):
    vals = [r[key] for r in runs]
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lrs", type=float, nargs="+", default=[5e-5, 2e-4, 5e-4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--quick", action="store_true", help="1 epoch, 1 lr, 1 seed (smoke test)")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.lrs, args.seeds = 1, args.lrs[:1], args.seeds[:1]

    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    contamination = json.loads((NEWS / "contamination.json").read_text())
    labels = contamination["labels"]
    k = len(labels)
    train_rows, dev_rows, test_rows = read_jsonl("train"), read_jsonl("dev"), read_jsonl("test")
    results = {"labels": labels, "args": vars(args), "setups": {}}

    # --- baselines -----------------------------------------------------------
    majority = Counter(r["label_id"] for r in train_rows).most_common(1)[0][0]
    results["setups"]["majority"] = {"test": score_test([majority] * len(test_rows), test_rows, k)}

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    xtr = vec.fit_transform([r["text"] for r in train_rows])
    xdv, xte = vec.transform([r["text"] for r in dev_rows]), vec.transform([r["text"] for r in test_rows])
    ytr, ydv = [r["label_id"] for r in train_rows], [r["label_id"] for r in dev_rows]
    best_c, best_f1 = None, -1
    for c in (0.3, 1, 3, 10, 30, 100):
        clf = LogisticRegression(C=c, max_iter=3000).fit(xtr, ytr)
        f1 = scores(clf.predict(xdv).tolist(), ydv, k)["macro_f1"]
        if f1 > best_f1:
            best_c, best_f1 = c, f1
    clf = LogisticRegression(C=best_c, max_iter=3000).fit(xtr, ytr)
    results["setups"]["tfidf_logreg"] = {"C": best_c, "dev_macro_f1": best_f1,
                                         "test": score_test(clf.predict(xte).tolist(), test_rows, k)}
    for name in ("majority", "tfidf_logreg"):
        t = results["setups"][name]["test"]
        print(f"{name:<22} test acc {t['accuracy']:.3f} | macro-F1 {t['macro_f1']:.3f}", flush=True)

    # --- fine-tuned GPTs -------------------------------------------------------
    ckpt = torch.load(PRETRAINED, map_location="cpu")
    cfg, state = ckpt["config"], ckpt["model_state_dict"]
    max_len = cfg["context_length"]
    dev_loader = make_loader(NEWS / "dev.jsonl", max_len, 32, shuffle=False)
    test_loader = make_loader(NEWS / "test.jsonl", max_len, 32, shuffle=False)

    def run(setup, lr, seed):
        torch.manual_seed(seed)
        pretrained, trainable = {
            "random_init_all": (None, "all"),
            "pretrained_last_block": (state, "last_block"),
            "pretrained_all": (state, "all"),
        }[setup]
        model = build_classifier(cfg, k, pretrained, trainable)
        train_loader = make_loader(NEWS / "train.jsonl", max_len, args.batch_size, shuffle=True, seed=seed)
        t0 = time.time()
        best, history = train_classifier(model, train_loader, dev_loader, device, k, lr, args.epochs)
        test = score_test(predict(model, test_loader, device)[0], test_rows, k)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return model, {"lr": lr, "seed": seed, "best_epoch": best["epoch"],
                       "dev_macro_f1": best["macro_f1"], "dev_accuracy": best["accuracy"],
                       "test": test, "history": history, "trainable_params": trainable_params,
                       "minutes": round((time.time() - t0) / 60, 2)}

    overall_best = (-1.0, None)
    for setup in ("random_init_all", "pretrained_last_block", "pretrained_all"):
        print(f"\n=== {setup}", flush=True)
        sweep = []
        for lr in args.lrs:
            print(f"  lr {lr:g}, seed {args.seeds[0]}", flush=True)
            model, r = run(setup, lr, args.seeds[0])
            sweep.append(r)
            if r["dev_macro_f1"] > overall_best[0]:
                overall_best = (r["dev_macro_f1"], setup)
                torch.save({"model_state_dict": model.state_dict(), "config": cfg, "labels": labels,
                            "setup": setup, "lr": lr}, OUT / "best_classifier.pt")
        best_lr = max(sweep, key=lambda r: r["dev_macro_f1"])["lr"]
        runs = [r for r in sweep if r["lr"] == best_lr]
        for seed in args.seeds[1:]:
            print(f"  lr {best_lr:g}, seed {seed}", flush=True)
            runs.append(run(setup, best_lr, seed)[1])
        summary = {key: mean_std([r["test"] for r in runs], key)
                   for key in ("accuracy", "macro_f1", "clean_accuracy", "clean_macro_f1")}
        results["setups"][setup] = {"best_lr": best_lr, "sweep": sweep, "runs": runs, "test": summary,
                                    "trainable_params": runs[0]["trainable_params"]}
        print(f"  -> best lr {best_lr:g} | test acc {summary['accuracy']['mean']:.3f} "
              f"± {summary['accuracy']['std']:.3f} | macro-F1 {summary['macro_f1']['mean']:.3f} "
              f"± {summary['macro_f1']['std']:.3f}", flush=True)
        (OUT / "results.json").write_text(json.dumps(results, indent=2))

    results["best_dev_setup"] = overall_best[1]
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print("\nsaved checkpoints/news-classifier/results.json", flush=True)


if __name__ == "__main__":
    main()
