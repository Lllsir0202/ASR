import os

speakers = set(os.listdir('./test/test'))
# print(speakers)

# 将所有在speakers集合中的txt写入text
text_dir = "../transcript/"

output_dir = "../"

if os.path.exists(os.path.join(output_dir, 'text')):
    with open(os.path.join(output_dir, 'text'), 'w', encoding='utf-8'):
        pass

for file in os.listdir(text_dir):
    # print(file)
    speaker = file.split('.',1)[0]
    # print(speaker)
    if speaker in speakers:
        with open(os.path.join(text_dir, file), 'r', encoding='utf-8') as txt:
            with open(os.path.join(output_dir, 'text'), 'a', encoding='utf-8') as f:
                for line in txt:
                    wav, text = line.strip().split('\t', 1)
                    wav = wav.split('.', 1)[0]
                    # print(wav, text)
                    f.write(wav)
                    f.write(" ")
                    f.write(text)
                    f.write("\n")