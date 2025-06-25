import os

def convert_data_format(input_base_path, output_base_path, split):
    tsv_file = os.path.join(input_base_path, f"{split}.tsv")
    zh_file = os.path.join(input_base_path, f"{split}.zh")
    output_file = os.path.join(output_base_path, f"{split}_data_no_timestamps.txt")

    print(f"Converting {tsv_file} and {zh_file} to {output_file}...")

    audio_paths = []
    with open(tsv_file, 'r', encoding='utf-8') as f_tsv:
        # Explicitly skip the first line (header or first data line if no header)
        next(f_tsv, None)
        
        # Assuming the format is: ID \t VIDEO_PATH \t AUDIO_PATH \t AUDIO_LEN \t VIDEO_LEN
        # And audio_path is the 3rd column (index 2)
        for line in f_tsv:
            parts = line.strip().split('\t')
            if len(parts) > 2: # Ensure it has enough columns
                audio_paths.append(parts[2].strip()) # Assuming 3rd column is audio path
            else:
                print(f"Skipping malformed TSV line: {line.strip()}")

    transcriptions = []
    with open(zh_file, 'r', encoding='utf-8') as f_zh:
        for line in f_zh:
            transcriptions.append(line.strip())

    if len(audio_paths) != len(transcriptions):
        print(f"Warning: Mismatch in number of lines between TSV ({len(audio_paths)}) and ZH ({len(transcriptions)}) for split {split}. This might lead to incorrect data alignment.")
        # We will proceed, but this is a potential issue to investigate if results are bad.

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(min(len(audio_paths), len(transcriptions))):
            f_out.write(f"{audio_paths[i]}\t{transcriptions[i]}\n")
    
    print(f"Conversion for {split} complete. Output saved to {output_file}")


if __name__ == "__main__":
    input_base_path = "/home/liuhaoze/liuhaoze_space/Code/whisper-flamingo/muavic/zh/muavic_normalized/"
    output_base_path = "./" # Current directory (whisper-finetuning)

    # Create the output directory if it doesn't exist
    os.makedirs(output_base_path, exist_ok=True)

    splits = ["train", "valid", "test"] # Assuming these are the splits you want to process
    # If you have test2, you can add it here too: splits = ["train", "valid", "test", "test2"]

    for split in splits:
        convert_data_format(input_base_path, output_base_path, split) 