import json
from pathlib import Path
import matplotlib.pyplot as plt

# Change this to your actual output dir
run_dir = Path("/Data/Laiba-Maddy-Project/Project2/outputs/ltx_reverse_smoke")

# Try to find W&B history-like files
candidate_files = list(run_dir.rglob("wandb-history.jsonl")) + list(run_dir.rglob("wandb-history.json"))

if not candidate_files:
    raise FileNotFoundError("Could not find wandb history file under output dir.")

history_file = candidate_files[0]
print(f"Using history file: {history_file}")

steps = []
losses = []
lrs = []

if history_file.suffix == ".jsonl":
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            step = row.get("_step")
            loss = row.get("train/loss", row.get("loss"))
            lr = row.get("lr", row.get("learning_rate"))

            if step is not None:
                if loss is not None:
                    steps.append(step)
                    losses.append(loss)
                if lr is not None and len(lrs) < len(steps):
                    lrs.append(lr)
else:
    with open(history_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for row in data:
        step = row.get("_step")
        loss = row.get("train/loss", row.get("loss"))
        lr = row.get("lr", row.get("learning_rate"))

        if step is not None and loss is not None:
            steps.append(step)
            losses.append(loss)
        if step is not None and lr is not None:
            lrs.append(lr)

# Loss plot
plt.figure(figsize=(8, 5))
plt.plot(steps[:len(losses)], losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.tight_layout()
plt.savefig(run_dir / "loss_curve.png", dpi=200)
plt.close()

# LR plot
if lrs:
    lr_steps = steps[:len(lrs)]
    plt.figure(figsize=(8, 5))
    plt.plot(lr_steps, lrs)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.tight_layout()
    plt.savefig(run_dir / "lr_curve.png", dpi=200)
    plt.close()

print("Saved plots.")