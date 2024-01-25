import numpy as np

import cv2
from moviepy.editor import *

from lip.extract import LipExtractor
from lip.feature import LipEmbedding

path_vid = "/home/data2/kbh/LRS3/test/0Fi83BHQsMA/00002.mp4"

video = VideoFileClip(path_vid)
print(video.fps)
fps = video.fps
print(video.reader.nframes)

audio = video.audio
print(audio.fps)
vid = []
for frame in video.iter_frames() :
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vid.append(frame)
vid = np.array(vid)
print(vid.shape)

print(vid.shape)
extractor = LipExtractor("cuda:1")
seq = extractor(vid)
print(seq.shape)

m = LipEmbedding()