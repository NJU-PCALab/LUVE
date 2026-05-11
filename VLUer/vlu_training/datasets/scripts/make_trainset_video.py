import os
from PIL import Image
import pickle
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import csv

import models
import core

# config
video_dir = "./dataset/video"
base_dir = './dataset'
down_scales = [1.5, 2]
world_size = 2 # GPU num
max_sample = 20000
processed_info_path = f'{base_dir}/processed_info.txt'
error_csv_path = f'{base_dir}/failed_videos.csv'
sd_ckpt = 'Wan2.1_VAE.pth' # TODO
file_sort = True


os.makedirs(f'{base_dir}/HR_latent', exist_ok=True)
for ds in down_scales:
    os.makedirs(f'{base_dir}/LR_latent/X{ds}', exist_ok=True)
if not os.path.exists(error_csv_path):
    with open(error_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_name', 'num_frames', 'error'])
if os.path.exists(processed_info_path):
    with open(processed_info_path, 'r') as f:
        processed_info = set(line.strip() for line in f)
else:
    processed_info = set()

def video_generator(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if file_sort:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return [os.path.join(folder, f) for f in files]

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames

def process_video(file_path, vae, device):
    video_name = os.path.basename(file_path)
    idx = os.path.splitext(video_name)[0]
    if idx in processed_info:
        print(f"[skip] Already processed {video_name}")
        return
    try:
        frames = load_video_frames(file_path)
        if len(frames) == 0:
            print(f"No frames found in video: {video_name}")
            return
        video_tensor = torch.stack([transforms.ToTensor()(f) for f in frames], dim=1).unsqueeze(0).to(device)
        print(f"[GPU{device.index}] Loaded {video_name} with {video_tensor.shape}")
        # HR latent
        with torch.no_grad():
            hr_latent = vae.encode((video_tensor - 0.5) * 2)
        np.save(f'{base_dir}/HR_latent/{idx}.npy', hr_latent[0].permute(1,2,3,0).cpu().numpy())
        del hr_latent

        # LR latent
        for ds in down_scales:
            lr_frames = [core.imresize(video_tensor[:, :, t, :, :], sizes=(int(video_tensor.shape[3] // ds), int(video_tensor.shape[4] // ds))) for t in range(video_tensor.shape[2])]
            lr = torch.stack(lr_frames, dim=2).to(device)
            lr = (lr * 255).clip(0, 255).to(torch.uint8).float() / 255
            with torch.no_grad():
                lr_latent = vae.encode((lr - 0.5) * 2)
            np.save(f'{base_dir}/LR_latent/X{ds}/{idx}.npy', lr_latent[0].permute(1,2,3,0).cpu().numpy())
            del lr_latent, lr

        del video_tensor
        print(f"[GPU{device.index}] Processed {video_name}")
        with open(processed_info_path, 'a') as f:
            f.write(f"{idx}\n")

    except Exception as e:
        print(f"[error] processing {video_name}: {e}")
        with open(error_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([video_name, len(frames), str(e)])
        torch.cuda.empty_cache()

def worker(rank, video_list):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    vae = models.vae.WanVAE(vae_pth=sd_ckpt, device=device)
    for _, file_path in enumerate(video_list[rank]):
        process_video(file_path, vae, device)

def main():
    video_files = video_generator(video_dir)
    chunks = [video_files[i::world_size] for i in range(world_size)]
    mp.spawn(worker, args=(chunks,), nprocs=world_size)

if __name__ == "__main__":
    main()
