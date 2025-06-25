import os
from pathlib import Path

def convert_to_srt(input_txt_file, output_srt_dir):
    Path(output_srt_dir).mkdir(parents=True, exist_ok=True)
    
    with open(input_txt_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                audio_path = parts[0]
                transcription = parts[1]
                
                # Extract filename without extension from audio_path
                # This should match the filenames in transcribed_results
                srt_filename = Path(audio_path).stem + ".srt"
                output_srt_path = Path(output_srt_dir) / srt_filename
                
                # Write a simple SRT file. Timestamps are dummy as calculate_metric.py only uses text.
                with open(output_srt_path, 'w', encoding='utf-8') as f_out:
                    f_out.write("1\n")
                    f_out.write("00:00:00,000 --> 00:00:01,000\n") # Dummy timestamp
                    f_out.write(f"{transcription}\n\n")
            else:
                print(f"Warning: Skipping malformed line in input TXT file: {line.strip()}")
    print(f"Conversion complete. SRT files saved to {output_srt_dir}")

if __name__ == "__main__":
    input_file = "/home/liuhaoze/liuhaoze_space/Code/whisper-finetuning/child_dev.txt"
    output_dir = "/home/liuhaoze/liuhaoze_space/Code/whisper-finetuning/ground_truth_child_srt_files_dev"
    convert_to_srt(input_file, output_dir) 