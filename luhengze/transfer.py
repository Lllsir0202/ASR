import os
folders=os.listdir("./wav/train")

# wav.scp:
# each line: wav_id \t abs path
# wav_scp=open("./wav.scp",'w')

# text
# each line: wav_id \t transcript
# text=open("./text",'w')
data_list=open("./data_train.list",'w')
folders=folders[1:]
print(folders)
for folder in folders:
    wavs=os.listdir(f"./wav/train/{folder}")
    trans_f=open(f"./transcript/{folder}.txt",encoding='utf-8')
    trans_list=[]
    for it in trans_f:
        trans_list.append(it)
    print(wavs)
    print(trans_list)
    assert len(wavs)==len(trans_list)
    wavs=sorted(wavs)
    for wav, trans in zip(wavs,trans_list):
        wav_name,trans_text=trans.split('\t')
        trans_text=trans_text.strip('\n')
        # assert wav_name==wav # 不一定
        abs_fp=os.path.abspath(f"./wav/train/{folder}/{wav}")
        # wav_scp.write(f"{wav[:-4]}\t{abs_fp}\n")
        # text.write(f"{wav[:-4]}\t{trans_text}\n")
        data_list.write(f'{{"key": "{wav[:-4]}", "wav": "{abs_fp}", "txt": "{trans_text}"}}\n')
        
    # abs_fp=os.path.abspath(f"./wav/test/{folder}")
    # print(wavs)