import os,glob 
from tqdm import tqdm
import wave

root =  "/home/data2/kbh/AI_HUB_speech/KsponSpeech_01/"
root_out = "/home/data2/kbh/KsponSpeech_01_WAV"
list_target = [ x for x in glob.glob(os.path.join(root,"**","*.pcm"),recursive=True)]

os.makedirs(root_out,exist_ok=True)

for path_pcm in tqdm(list_target) : 
    with open(path_pcm, 'rb') as f:
        pcm = f.read()

    name_pcm = path_pcm.split('/')[-1]
    id_pcm = name_pcm.split('.')[0]

    with wave.open(os.path.join(root_out,id_pcm+".wav"), 'wb') as wavfile:
        wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        wavfile.writeframes(pcm)


    