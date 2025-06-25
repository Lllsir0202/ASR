import argparse
import whisper
import torch
from pathlib import Path
from tqdm import tqdm
import json
import jiwer
from whisper.tokenizer import get_tokenizer

# We need the KNN_Dstore class from our main training script
from run_finetuning import KNN_Dstore

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Whisper model with optional k-NN augmentation.")
    # Model and Data paths
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the fine-tuned model checkpoint (.pt file)."
    )
    parser.add_argument(
        "--eval-json", type=str, required=True, help="Path to the JSON file containing evaluation data."
    )
    parser.add_argument(
        "--model-name", type=str, default="tiny", help="Name of the base Whisper model."
    )
    # k-NN arguments
    parser.add_argument(
        "--use-knn", action="store_true", help="Enable k-NN augmented decoding."
    )
    parser.add_argument(
        "--dstore-dir", type=str, default="dstore", help="Directory of the k-NN datastore."
    )
    parser.add_argument(
        "--knn-k", type=int, default=16, help="The 'k' in k-NN."
    )
    parser.add_argument(
        "--knn-lambda", type=float, default=0.5, help="Interpolation factor for k-NN probabilities."
    )
    # General arguments
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for evaluation."
    )
    parser.add_argument(
        "--language", type=str, default="Chinese", help="Language for decoding."
    )
    parser.add_argument(
        "--beam-size", type=int, default=5, help="Beam size for standard decoding (when not using k-NN)."
    )
    return parser

@torch.no_grad()
def decode_with_knn(model: whisper.Whisper, dstore: KNN_Dstore, audio: torch.Tensor, tokenizer: whisper.tokenizer.Tokenizer, args: argparse.Namespace):
    """
    Performs greedy decoding, augmenting the probability distribution with k-NN at each step.
    """
    sot_token = tokenizer.sot
    eot_token = tokenizer.eot
    
    # Start with the start-of-transcript token
    tokens = torch.tensor([[sot_token]], device=args.device)

    # Encode the audio
    audio_features = model.embed_audio(audio.unsqueeze(0))

    # The loop to generate tokens one by one
    for _ in range(model.dims.n_text_ctx // 2):
        # --- Model's forward pass (manual, step-by-step) ---
        # We do this manually to get the hidden state before the final layer norm
        current_tokens = tokens
        hidden_states = model.decoder.token_embedding(current_tokens) + model.decoder.positional_embedding[:current_tokens.shape[-1]]
        hidden_states = hidden_states.to(audio_features.dtype)
        for block in model.decoder.blocks:
            hidden_states = block(hidden_states, audio_features, mask=model.decoder.mask)
        hidden_states_norm = model.decoder.ln(hidden_states)

        # Get the hidden state for the *last* token, which is our query for k-NN
        query_vector = hidden_states_norm[:, -1:, :] # Shape: [1, 1, dim]

        # --- Get Probabilities ---
        # 1. Model's probabilities
        logits = (hidden_states_norm.float() @ model.decoder.token_embedding.weight.t())[:, -1, :]
        model_probs = torch.nn.functional.softmax(logits, dim=-1)

        # 2. k-NN probabilities
        knn_probs = dstore.get_knn_probs(query_vector, k=args.knn_k).squeeze(1) # Squeeze seq_len dim

        # --- Interpolate ---
        combined_probs = (1 - args.knn_lambda) * model_probs + args.knn_lambda * knn_probs
        
        # --- Select next token (greedy) ---
        next_token = combined_probs.argmax(dim=-1).unsqueeze(0)
        tokens = torch.cat([tokens, next_token], dim=1)

        # Check for end-of-text token
        if next_token.item() == eot_token:
            break
            
    # Decode the final sequence of tokens
    text = tokenizer.decode(tokens[0].cpu().numpy())
    return text


def main():
    args = get_parser().parse_args()
    
    # --- Load Model ---
    print(f"Loading model from {args.model_path}...")
    model = whisper.load_model(args.model_name, device=args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Load k-NN Datastore (if requested) ---
    dstore = None
    if args.use_knn:
        print(f"Loading k-NN datastore from {args.dstore_dir}...")
        dstore = KNN_Dstore(args.dstore_dir, model.dims.n_vocab, device=args.device)
        print("Datastore loaded.")

    # --- Load Data and Tokenizer ---
    print(f"Loading evaluation data from {args.eval_json}...")
    with open(args.eval_json, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    tokenizer = get_tokenizer(multilingual=".en" not in args.model_name, task="transcribe", language=args.language.lower())
    
    predictions = []
    references = []

    # --- Main Evaluation Loop ---
    for item in tqdm(eval_data, desc="Evaluating"):
        audio_path = item['audio_path']
        reference_text = item['text']
        
        # Load and process audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        if args.use_knn:
            # Use our custom decoding function
            predicted_text = decode_with_knn(model, dstore, audio, tokenizer, args)
        else:
            # Use standard beam search decoding
            options = whisper.DecodingOptions(language=args.language, without_timestamps=True, beam_size=args.beam_size)
            result = model.transcribe(audio_path, **vars(options))
            predicted_text = result['text']
        
        predictions.append(predicted_text)
        references.append(reference_text)

    # --- Calculate and Print Metrics ---
    print("\nCalculating metrics...")
    # For Chinese, it's better to remove spaces for CER calculation
    transformation = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ToLowerCase(),
    ])

    wer = jiwer.wer(references, predictions, truth_transform=transformation, hypothesis_transform=transformation)
    
    # CER transform: remove spaces between characters
    cer_transformation = jiwer.Compose([
        transformation,
        jiwer.RemoveEmptyStrings(),
        jiwer.Substitute(substitutions={" ": ""})
    ])
    cer = jiwer.cer(references, predictions, truth_transform=cer_transformation, hypothesis_transform=cer_transformation)

    print(f"\n--- Evaluation Results (Use k-NN: {args.use_knn}) ---")
    print(f"Word Error Rate (WER): {wer:.4f}")
    print(f"Character Error Rate (CER): {cer:.4f}")
    print("--------------------------")

if __name__ == "__main__":
    main() 