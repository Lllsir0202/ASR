import argparse
from pathlib import Path
from typing import Iterator, Union

import torch
import whisper
from tqdm import tqdm
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from whisper.utils import get_writer
import numpy as np
import re

# --- NEW: Import k-NN dependencies ---
from run_finetuning import KNN_Dstore


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe audio files with a Whisper model")
    # --- Modified and New Arguments ---
    parser.add_argument("--model", default="large", help="Name of the base Whisper model architecture (e.g., 'tiny', 'base').")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the fine-tuned model checkpoint (.pt file).")
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Path to directory containing audio files to transcribe. Either this or --data-file is required.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to a text file containing audio paths, one per line.",
    )
    parser.add_argument("--save-dir", type=str, default="output", help="Path to directory to save transcribed results.")
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="Language of the data.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference.")
    parser.add_argument(
        "--task", type=str, default="transcribe", choices=["transcribe", "translate"],
        help="Whether to perform speech recognition ('transcribe') or translation ('translate')."
    )
    # k-NN arguments
    parser.add_argument("--use-knn", action="store_true", help="Enable k-NN augmented decoding.")
    parser.add_argument("--dstore-dir", type=str, default="dstore", help="Directory of the k-NN datastore.")
    parser.add_argument("--knn-k", type=int, default=16, help="The 'k' in k-NN.")
    parser.add_argument("--knn-lambda", type=float, default=0.5, help="Interpolation factor for k-NN probabilities.")
    return parser


# --- NEW: k-NN Decoding Function (copied from evaluation.py) ---
@torch.no_grad()
def decode_with_knn(model: whisper.Whisper, dstore: KNN_Dstore, audio: np.ndarray, tokenizer: whisper.tokenizer.Tokenizer, args: argparse.Namespace):
    sot_token = tokenizer.sot
    eot_token = tokenizer.eot
    tokens = torch.tensor([[sot_token]], device=args.device)
    
    # --- FIXED: Manually perform spectrogram conversion ---
    # 1. Compute the log-Mel spectrogram from the raw audio numpy array
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    
    # 2. Get audio features from the encoder
    audio_features = model.embed_audio(mel.unsqueeze(0)) # Add batch dimension

    for _ in range(model.dims.n_text_ctx // 2):
        current_tokens = tokens
        hidden_states = model.decoder.token_embedding(current_tokens) + model.decoder.positional_embedding[:current_tokens.shape[-1]]
        hidden_states = hidden_states.to(audio_features.dtype)
        for block in model.decoder.blocks:
            hidden_states = block(hidden_states, audio_features, mask=model.decoder.mask)
        hidden_states_norm = model.decoder.ln(hidden_states)
        query_vector = hidden_states_norm[:, -1:, :]
        logits = (hidden_states_norm.float() @ model.decoder.token_embedding.weight.t())[:, -1, :]
        model_probs = torch.nn.functional.softmax(logits, dim=-1)
        knn_probs = dstore.get_knn_probs(query_vector, k=args.knn_k).squeeze(1)
        combined_probs = (1 - args.knn_lambda) * model_probs + args.knn_lambda * knn_probs
        next_token = combined_probs.argmax(dim=-1).unsqueeze(0)
        tokens = torch.cat([tokens, next_token], dim=1)
        if next_token.item() == eot_token:
            break
    
    # --- FINAL FIX: Remove all special tokens using regex ---
    text = tokenizer.decode(tokens[0].cpu().numpy())
    # Remove all <|...|> patterns
    text = re.sub(r'<\|.*?\|>', '', text)
    text = text.strip()
    return text


def main():
    args = get_parser().parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # --- Load Model (handles both original and fine-tuned) ---
    print(f"Loading base model: {args.model}")
    model = whisper.load_model(args.model, args.device)
    if args.model_path:
        print(f"Applying fine-tuned weights from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Load Tokenizer and k-NN Datastore (if requested) ---
    tokenizer = get_tokenizer(multilingual=".en" not in args.model, task=args.task, language=args.language.lower())
    dstore = None
    if args.use_knn:
        print(f"Loading k-NN datastore from {args.dstore_dir}...")
        if not Path(args.dstore_dir).exists():
            raise FileNotFoundError(f"Datastore directory not found: {args.dstore_dir}")
        dstore = KNN_Dstore(args.dstore_dir, model.dims.n_vocab, device=args.device)
        print("Datastore loaded.")

    writer = get_writer("srt", args.save_dir)

    # --- Load Audio File List ---
    if args.data_file:
        audio_files_to_process = []
        with open(args.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                audio_path = line.strip().split('\t')[0]
                if Path(audio_path).exists():
                    audio_files_to_process.append(audio_path)
                else:
                    print(f"Warning: Skipping non-existent path: {audio_path}")
    elif args.audio_dir:
        audio_files_to_process = list(Path(args.audio_dir).iterdir())
    else:
        raise ValueError("Either --audio-dir or --data-file must be provided.")

    # --- Main Transcription Loop ---
    for audio_path_str in tqdm(audio_files_to_process, desc="Transcribing"):
        audio_path = Path(audio_path_str)
        
        if args.use_knn:
            audio = whisper.load_audio(str(audio_path))
            audio = whisper.pad_or_trim(audio)
            predicted_text = decode_with_knn(model, dstore, audio, tokenizer, args)
            result = {'text': predicted_text, 'segments': [{'text': predicted_text, 'start': 0, 'end': 0}]}
        else:
            result = model.transcribe(task=args.task, audio=str(audio_path), language=args.language)
        
        writer(result, str(audio_path))

    print(f"\nTranscription complete. Results saved in: {args.save_dir}")


if __name__ == "__main__":
    main()
