import json
import shutil
import subprocess
from pathlib import Path

# =========================
# Paths
# =========================
train_json = Path("/Data/Laiba-Maddy-Project/Project2/Dataset/something_something_v2/labels/train.json")
src_video_dir = Path("/Data/Laiba-Maddy-Project/Project2/Dataset/something_something_v2/20bn-something-something-v2")
out_root = Path("/Data/Laiba-Maddy-Project/Project2/Dataset/dataset2")
out_video_dir = out_root / "videos"
videos_txt = out_root / "videos.txt"
prompts_txt = out_root / "prompts.txt"

# =========================
# Target templates
# =========================
target_templates = {
    "Dropping [something] onto [something] ",
	"Dropping [something] next to [something]",
     "Dropping [something] into [something]",
	"Dropping [something] in front of [something]" ,
	 "[Something] falling like a rock",
	"[Something] falling like a feather or paper "
}

# =========================
# Helpers
# =========================
def find_video_file(video_id: str, src_dir: Path) -> Path | None:
    """
    Try common video extensions.
    """
    for ext in [".webm", ".mp4", ".avi", ".mov", ".mkv"]:
        p = src_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def reverse_video_ffmpeg(infile: Path, outfile: Path):
    """
    Reverse video frames only. something-something videos are usually fine without audio.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(infile),
        "-vf", "reverse",
        "-an",
        str(outfile)
    ]
    subprocess.run(cmd, check=True)


# =========================
# Main
# =========================
def main():
    out_video_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(train_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Keep entries in the same order as train.json
    selected = [item for item in data if item.get("template") in target_templates]

    original_paths = []
    original_prompts = []

    # Copy original videos and collect aligned paths/prompts
    for item in selected:
        video_id = item["id"]
        prompt = item["label"]

        src_video = find_video_file(video_id, src_video_dir)
        if src_video is None:
            print(f"[WARN] Video not found for id={video_id}")
            continue

        dst_video = out_video_dir / src_video.name
        if not dst_video.exists():
            shutil.copy2(src_video, dst_video)

        original_paths.append(str(dst_video))
        original_prompts.append(prompt)

    reversed_paths = []
    reversed_prompts = []

    # Reverse copied videos
    for path_str, prompt in zip(original_paths, original_prompts):
        original_path = Path(path_str)
        rev_name = f"rev_{original_path.stem}{original_path.suffix}"
        rev_path = out_video_dir / rev_name

        if not rev_path.exists():
            reverse_video_ffmpeg(original_path, rev_path)

        reversed_paths.append(str(rev_path))
        reversed_prompts.append(f"<REV> {prompt}")

    # Final aligned outputs:
    # originals first, then reversed in same sequence
    final_video_paths = original_paths + reversed_paths
    final_prompts = original_prompts + reversed_prompts

    with open(videos_txt, "w", encoding="utf-8") as f:
        for p in final_video_paths:
            f.write(p + "\n")

    with open(prompts_txt, "w", encoding="utf-8") as f:
        for p in final_prompts:
            f.write(p + "\n")

    print(f"Done.")
    print(f"Original selected videos: {len(original_paths)}")
    print(f"Reversed videos created: {len(reversed_paths)}")
    print(f"videos.txt: {videos_txt}")
    print(f"prompts.txt: {prompts_txt}")


if __name__ == "__main__":
    main()