import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import numpy as np
from argparse import ArgumentParser


class VideoDataset(Dataset):
    """Read videos from a directory for feature extraction."""
    def __init__(self, videos_dir, video_names):
        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_path = os.path.join(self.videos_dir, video_name)
        video_data = skvideo.io.vread(video_path)

        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        return transformed_video


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction."""
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling."""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)


def get_features(video_data, frame_batch_size=64, device='cuda'):
    """Feature extraction."""
    extractor = ResNet50().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
        while frame_end <= video_length:
            batch = video_data[frame_start:frame_end].to(device)
            features_mean, features_std = extractor(batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        if frame_start < video_length:
            last_batch = video_data[frame_start:video_length].to(device)
            features_mean, features_std = extractor(last_batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)

    return torch.cat((output1, output2), 1).squeeze()


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50"')
    parser.add_argument('--videos_dir', default="C:/D/VQA/IC/savoias/video/savoias_i2vgen/", help='Path to the directory containing MP4 videos.')
    parser.add_argument('--features_dir', default="C:/D/VQA/IC/savoias/video/motion_featuresVgen/", help='Path to the directory where features will be saved.')
    parser.add_argument('--frame_batch_size', type=int, default=1, help='Frame batch size for feature extraction.')
    parser.add_argument('--disable_gpu', action='store_true', help='Flag to disable GPU.')
    args = parser.parse_args()

    if not os.path.exists(args.features_dir):
        os.makedirs(args.features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    video_names = [f for f in os.listdir(args.videos_dir) if f.endswith('.mp4')]
    dataset = VideoDataset(args.videos_dir, video_names)

    for video_name in video_names:
        video_data = dataset[video_names.index(video_name)]
        print(f'Processing video: {video_name}, length: {video_data.shape[0]} frames')
        features = get_features(video_data, args.frame_batch_size, device)
        output_path = os.path.join(args.features_dir, os.path.splitext(video_name)[0] + '.npy')
        np.save(output_path, features.cpu().numpy())
        print(f'Features saved to: {output_path}')
