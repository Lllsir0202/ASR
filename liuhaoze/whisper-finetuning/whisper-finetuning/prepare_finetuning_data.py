import argparse
import string

def get_punctuation_set():
    """
    Returns a comprehensive set of Chinese and English punctuation.
    """
    # Standard English punctuation
    en_punct = string.punctuation
    # A comprehensive set of Chinese punctuation
    zh_punct = "＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹒¸．！？｡。"
    return set(en_punct + zh_punct)

PUNCTUATION_TO_REMOVE = get_punctuation_set()

def clean_text(text: str) -> str:
    """
    Removes specified punctuation characters and all whitespace from a text string.
    """
    # Remove punctuation
    text_no_punct = "".join(char for char in text if char not in PUNCTUATION_TO_REMOVE)
    # Remove all types of whitespace (regular space, full-width space, tabs, newlines etc.)
    text_no_space = ''.join(text_no_punct.split())
    return text_no_space

def main():
    """
    Reads a .tsv file and a .el file, cleans the transcriptions,
    and combines them into a single .txt file with the format:
    <absolute_audio_path>\t<cleaned_transcription>
    """
    parser = argparse.ArgumentParser(description="Prepare data for whisper-finetuning.")
    parser.add_argument("--tsv_file", required=True, help="Path to the input .tsv file.")
    parser.add_argument("--el_file", required=True, help="Path to the input .el file (transcriptions).")
    parser.add_argument("--output_file", required=True, help="Path to the output .txt file.")
    args = parser.parse_args()

    try:
        with open(args.tsv_file, 'r', encoding='utf-8') as f_tsv, \
             open(args.el_file, 'r', encoding='utf-8') as f_el, \
             open(args.output_file, 'w', encoding='utf-8') as f_out:

            tsv_lines = f_tsv.readlines()
            el_lines = f_el.readlines()

            if len(tsv_lines) != len(el_lines):
                print(f"Warning: Mismatch in line numbers between {args.tsv_file} ({len(tsv_lines)} lines) and {args.el_file} ({len(el_lines)} lines).")

            print(f"Processing {min(len(tsv_lines), len(el_lines))} lines...")

            for tsv_line, el_line in zip(tsv_lines, el_lines):
                parts = tsv_line.strip().split('\t')
                if len(parts) < 3:
                    # This handles empty lines in the tsv file
                    continue
                
                audio_path = parts[2]
                transcription = el_line.strip()
                cleaned_transcription = clean_text(transcription)

                if audio_path and cleaned_transcription:
                    f_out.write(f"{audio_path}\t{cleaned_transcription}\n")

        print(f"Successfully created {args.output_file}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()