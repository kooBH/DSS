import librosa
import os,glob




list_target = [x for x in glob.glob("/home/data2/kbh/LGE/DESED_100000_v1/*.wav")]

print(len(list_target))


for i in list_target : 
    print(i)
    x,_ = librosa.load(i,sr=16000,mono=False)