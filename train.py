"""
Train script.
"""
import numpy as np
from core import multiscale_fft, get_scheduler, mean_std_loudness
import torch
import yaml 
from dataloader import get_data_loader
from tqdm import tqdm
from model import WTS
from nnAudio import Spectrogram
import datetime
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import os
from utils import set_seed
import soundfile as sf

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# general parameters
sr = config["common"]["sampling_rate"]
block_size = config["common"]["block_size"]
duration_secs = config["common"]["duration_secs"]
batch_size = config["train"]["batch_size"]
scales = config["train"]["scales"]
overlap = config["train"]["overlap"]
hidden_size = config["train"]["hidden_size"]
n_harmonic = config["train"]["n_harmonic"]
n_bands = config["train"]["n_bands"]
n_wavetables = config["train"]["n_wavetables"]
n_mfcc = config["train"]["n_mfcc"]
train_lr = config["train"]["start_lr"]
epochs = config["train"]["epochs"]
global_seed = config["common"]["global_seed"]

mean_loudness = config["data"]["mean_loudness"] 
std_loudness = config["data"]["std_loudness"]

print("""
======================
sr: {}
block_size: {}
duration_secs: {}
batch_size: {}
scales: {}
overlap: {}
hidden_size: {}
n_harmonic: {}
n_bands: {}
n_wavetables: {}
n_mfcc: {}
train_lr: {}
global_seed:{}
======================
""".format(sr, block_size, duration_secs, batch_size, scales, overlap,
           hidden_size, n_harmonic, n_bands, n_wavetables, n_mfcc, train_lr, global_seed))

set_seed(global_seed)

model = WTS(hidden_size=hidden_size, n_harmonic=n_harmonic, n_bands=n_bands, sampling_rate=sr,
            block_size=block_size, n_wavetables=n_wavetables, mode="wavetable", 
            duration_secs=duration_secs)
model.cuda()
opt = torch.optim.Adam(model.parameters(), lr=train_lr)
spec = Spectrogram.MFCC(sr=sr, n_mfcc=n_mfcc)

# both values are pre-computed from the train setm, this only applied to v1
#mean_loudness, std_loudness = -39.74668743704927, 54.19612404969509
mean_loudness, std_loudness = -45.90259362462579, 83.69534087643832

train_dl = get_data_loader(config, mode="train", batch_size=batch_size)
valid_dl = get_data_loader(config, mode="valid", batch_size=batch_size)

#mean_loudness, std_loudness = mean_std_loudness(train_dl)
print(f"mean loudness: {mean_loudness}")
print(f"std loudness: {std_loudness}")
config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness


# for now the scheduler is not used
schedule = get_scheduler(
    len(train_dl),
    config["train"]["start_lr"],
    config["train"]["stop_lr"],
    config["train"]["decay_over"],
)

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/' + current_time +'/train'
model_dir = "outputs/" + current_time
os.makedirs(model_dir,exist_ok=False)

train_summary_writer = SummaryWriter(train_log_dir)
with open("config_"+ current_time + ".yaml", "w") as out_config:
    yaml.safe_dump(config, out_config)
    

best_loss = float("inf")
mean_loss = 0
idx = 0
n_element = 0

for ep in tqdm(range(0, epochs)):
    for y, loudness, pitch in tqdm(train_dl):
        mfcc = spec(y)
        pitch, loudness = pitch.unsqueeze(-1).float(), loudness.unsqueeze(-1).float()
        loudness = (loudness - mean_loudness) / std_loudness 

        mfcc = mfcc.cuda()
        pitch = pitch.cuda()
        loudness = loudness.cuda()

        output = model(mfcc, pitch, loudness)
        
        ori_stft = multiscale_fft(
                    torch.tensor(y).squeeze(),
                    scales,
                    overlap,
                )
        rec_stft = multiscale_fft(
            output.squeeze(),
            scales,
            overlap,
        )

        loss = 0
        for s_x, s_y in zip(ori_stft, rec_stft):
            s_x = s_x.cuda()
            s_y = s_y.cuda()
            
            lin_loss = (s_x - s_y).abs().mean()
            loss += lin_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        train_summary_writer.add_scalar('loss', loss.item(), global_step=idx)
        idx += 1
        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element
        
       
    if ep % 3 == 0:
        if mean_loss < best_loss:
            best_loss = mean_loss            
            torch.save(model.state_dict(), model_dir + "/" + "model.pt")
        mean_loss = 0
        n_element = 0
        #audio = torch.cat([torch.tensor(y).cuda(), output], -1).reshape(-1).detach().cpu().numpy()

        #sf.write(
        #    os.path.join(model_dir, f"eval_epoch_{ep:06d}.wav"),
        #    audio,
        #    sr,
        #)
        
        
        