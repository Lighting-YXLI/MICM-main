import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np



class ic_dataset(Dataset):
    def __init__(self, txt_path, img_path, vid_path,transform=None):
        super(ic_dataset, self).__init__()
        self.txt_lines = self.readlines(txt_path)
        self.img_path = img_path
        self.vid_path = vid_path
        self.transform = transform
        self.img_info_list = self.parse_lines(self.txt_lines)


    def parse_lines(self, lines):
        image_info_list = []
        for idx, line in enumerate(lines):
            line_split = line.strip().split("\t")
            img_name = line_split[0]
            img_prom = line_split[1]
            if len(line_split) <3:
                print('error')# Print
            img_label = line_split[2]  # This is causing the IndexError
            image_info_list.append((img_name, img_prom, img_label))

        return image_info_list

    def readlines(self, txt_path):
        with open(txt_path, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()
        return lines

    def __getitem__(self, index):
        imgName, prompt, imgLabel = self.img_info_list[index]
        oriImgPath = os.path.join(self.img_path, imgName)
        vidname = imgName.split('.')[0]+'.npy'
        motionvpath = os.path.join(self.vid_path, vidname)
        video = np.load(motionvpath)
        img = Image.open(oriImgPath).convert("RGB")
        img = self.transform(img)
        if imgLabel == 'An advertisement for a car that is made of wood and has beans on the ground around it, with the number 1 next to it in grains of rice or some':
            print('error')
        label = torch.tensor(float(imgLabel))

        return img, video,label, prompt

    def __len__(self):
        return len(self.img_info_list)

    @staticmethod
    def load_video(video_path):
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=torch.float32).permute(2, 0, 1))
        cap.release()
        video_tensor = torch.stack(frames, dim=0)  # Shape: [T, C, H, W]
        return video_tensor

#if __name__ == "__main__":
#    txt_path = "C:/D/VQA/IC/mytestnew.txt"
#    img_path = "C:/D/VQA/IC/images/"

#    dataset = ic_dataset(txt_path, img_path, transform=transforms.ToTensor())
#    for idx in range(len(dataset)):
#        data = dataset[idx]
#        if data is None:
#            print(f"Skipping index {idx} due to errors.")
#        else:
#            img, label, prompt, entropy = data
#            print(f"Index {idx}: Label={label}, Prompt={prompt}")
