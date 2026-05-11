import os
import csv
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def get_video_info(path):
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    out = subprocess.check_output(probe_cmd).decode().strip().split("\n")
    width, height, nb_frames = map(int, out)
    return width, height, nb_frames


def process_single_video(in_path, out_path, width, height):
    if width < height:
        scale_filter = f"scale=1440:-1"
    else:
        scale_filter = f"scale=-1:1440"
    crop_filter = f"crop=1440:1440"
    vf = f"{scale_filter},{crop_filter}"
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-vf", vf,
        "-frames:v", "41",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_existing_mapping(csv_path):
    mapping = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                idx, in_path, out_path = row
                mapping[in_path] = int(idx)
    return mapping


def append_mapping(csv_path, idx, in_path, out_path):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([idx, in_path, out_path])


def process_videos(input_dir, output_dir, csv_path="mapping.csv"):
    os.makedirs(output_dir, exist_ok=True)
    exts = [".mp4", ".avi", ".mov", ".mkv"]
    finished = load_existing_mapping(csv_path)
    if finished:
        max_idx = max(finished.values())
    else:
        max_idx = 0

    futures = {}
    with ProcessPoolExecutor() as executor:
        with tqdm(desc="Processing", unit="video") as pbar:
            idx = max_idx + 1
            for root, _, files in os.walk(input_dir):
                for file in files:
                    in_path = os.path.join(root, file)
                    if not any(in_path.lower().endswith(ext) for ext in exts):
                        continue
                    if in_path in finished:
                        continue
                    try:
                        width, height, nb_frames = get_video_info(in_path)
                    except Exception as e:
                        print(f"[skip] {in_path}, fail to read info: {e}")
                        continue

                    if width > 1440 and height > 1440 and nb_frames >= 41:
                        out_path = os.path.join(output_dir, f"{idx:04d}.mp4")
                        futures[executor.submit(
                            process_single_video, in_path, out_path, width, height
                        )] = (idx, in_path, out_path)
                        idx += 1

                    done = [f for f in futures if f.done()]
                    for f in done:
                        try:
                            f.result()
                            task_idx, task_in, task_out = futures[f]
                            append_mapping(csv_path, task_idx, task_in, task_out)
                        except Exception as e:
                            print(f"[error] {futures[f][1]}: {e}")
                        del futures[f]
                        pbar.update(1)

            for f in as_completed(futures):
                try:
                    f.result()
                    task_idx, task_in, task_out = futures[f]
                    append_mapping(csv_path, task_idx, task_in, task_out)
                except Exception as e:
                    print(f"[error] {futures[f][1]}: {e}")
                pbar.update(1)


if __name__ == "__main__":
    input_dir = "./media"
    output_dir = "./dataset/video"
    csv_path = os.path.join(output_dir, "mapping.csv")
    process_videos(input_dir, output_dir, csv_path)