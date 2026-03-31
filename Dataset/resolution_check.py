from pathlib import Path
import shutil
import torchvision.io as io

# =========================
# Config
# =========================
dataset_root = Path("/Data/Laiba-Maddy-Project/Project2/Dataset/dataset2")
videos_txt = dataset_root / "videos.txt"
prompts_txt = dataset_root / "prompts.txt"

min_h = 224
min_w = 320

make_backup = True

# =========================
# Helpers
# =========================
def read_lines(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def write_lines(path: Path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

def get_video_hw(video_path: Path):
    """
    Returns (height, width).
    """
    video, _, _ = io.read_video(str(video_path), pts_unit="sec")
    # shape: T, H, W, C
    if video.numel() == 0:
        raise ValueError("Empty video")
    h, w = int(video.shape[1]), int(video.shape[2])
    return h, w

# =========================
# Main
# =========================
def main():
    if not videos_txt.exists():
        raise FileNotFoundError(f"Missing file: {videos_txt}")
    if not prompts_txt.exists():
        raise FileNotFoundError(f"Missing file: {prompts_txt}")

    videos = read_lines(videos_txt)
    prompts = read_lines(prompts_txt)

    if len(videos) != len(prompts):
        raise ValueError(
            f"Mismatch: videos.txt has {len(videos)} lines but prompts.txt has {len(prompts)} lines"
        )

    if make_backup:
        shutil.copy2(videos_txt, dataset_root / "videos_backup.txt")
        shutil.copy2(prompts_txt, dataset_root / "prompts_backup.txt")

    kept_videos = []
    kept_prompts = []

    removed_videos = []
    removed_prompts = []

    for idx, (video_line, prompt_line) in enumerate(zip(videos, prompts), start=1):
        video_path = Path(video_line)

        try:
            if not video_path.exists():
                print(f"[REMOVE] #{idx} missing file: {video_path}")
                removed_videos.append(video_line)
                removed_prompts.append(prompt_line)
                continue

            h, w = get_video_hw(video_path)

            if h < min_h or w < min_w:
                print(f"[REMOVE] #{idx} {video_path.name} -> ({h}, {w})")
                removed_videos.append(video_line)
                removed_prompts.append(prompt_line)
            else:
                kept_videos.append(video_line)
                kept_prompts.append(prompt_line)

        except Exception as e:
            print(f"[REMOVE] #{idx} {video_path.name} -> error reading video: {e}")
            removed_videos.append(video_line)
            removed_prompts.append(prompt_line)

    # Rewrite aligned files without disturbing relative order
    write_lines(videos_txt, kept_videos)
    write_lines(prompts_txt, kept_prompts)

    # Save removed entries for inspection
    write_lines(dataset_root / "removed_videos.txt", removed_videos)
    write_lines(dataset_root / "removed_prompts.txt", removed_prompts)

    print("\nDone.")
    print(f"Kept pairs   : {len(kept_videos)}")
    print(f"Removed pairs: {len(removed_videos)}")
    print(f"Updated: {videos_txt}")
    print(f"Updated: {prompts_txt}")
    print(f"Removed list: {dataset_root / 'removed_videos.txt'}")
    print(f"Removed list: {dataset_root / 'removed_prompts.txt'}")

if __name__ == "__main__":
    main()