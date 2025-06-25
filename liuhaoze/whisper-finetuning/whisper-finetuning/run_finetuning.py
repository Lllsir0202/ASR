import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import whisper
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from whisper import Whisper
from whisper.tokenizer import get_tokenizer

from dataloader import get_dataloader


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a Whisper model for ASR")
    # Dataloader-related arguments
    parser.add_argument(
        "--train-json",
        type=str,
        default=None,
        help="Path to a json file containing training data. Required for training.",
    )
    parser.add_argument(
        "--dev-json",
        type=str,
        default=None,
        help="Path to a json file containing development data. Required for training.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--dev-batch-size", type=int, default=16, help="Batch size for validation")
    parser.add_argument(
        "--no-timestamps-training",
        action="store_true",
        help="Always use the no-timestamps training mode",
    )
    parser.add_argument(
        "--prompt-use-rate",
        type=float,
        default=0.5,
        help="How often to use prompts for conditioning the generation",
    )
    parser.add_argument(
        "--no-timestamps-rate",
        type=float,
        default=0.5,
        help=(
            "How often to use the no-timestamps mode. Only used if --no-timestamps-training "
            "is NOT set"
        ),
    )

    # Training-related arguments
    parser.add_argument(
        "--save-dir", type=str, default="output", help="directory to save the model"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for training",
    )
    parser.add_argument(
        "--model",
        default="large",
        choices=whisper.available_models(),
        help="name of the Whisper model to use",
    )
    parser.add_argument("--train-only-decoder", action="store_true", help="train only the decoder")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument(
        "--accum-grad-steps",
        type=int,
        default=64,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=5000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Number of steps to evaluate the model",
    )
    parser.add_argument(
        "--save-all-checkpoints",
        action="store_true",
        help="Save all checkpoints instead of only the best and the last one",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Perform evaluation only, without training.",
    )
    parser.add_argument(
        "--use-adam-8bit",
        action="store_true",
        help="Use Adam 8bit optimizer for reduced VRAM usage.",
    )
    # KNN-related arguments
    parser.add_argument(
        "--build-dstore",
        action="store_true",
        help="Build the datastore for KNN.",
    )
    parser.add_argument(
        "--load-from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to load the model from.",
    )
    parser.add_argument(
        "--use-knn",
        action="store_true",
        help="Use KNN for evaluation.",
    )
    parser.add_argument(
        "--dstore-dir",
        type=str,
        default="dstore",
        help="directory to save/load the datastore.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=16,
        help="K for KNN.",
    )
    parser.add_argument(
        "--knn-lambda",
        type=float,
        default=0.5,
        help="Lambda for KNN interpolation.",
    )
    return parser


def train_step(
    model: Whisper,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    accum_grad_steps: int,
    train_only_decoder: bool,
    max_grad_norm: float,
) -> Tuple[float, Iterator]:
    model.train()
    total_loss = 0
    for _ in range(accum_grad_steps):
        x, y_in, y_out = next(train_iter)
        x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)

        if train_only_decoder:
            with torch.no_grad():
                audio_features = model.embed_audio(x)
        else:
            audio_features = model.embed_audio(x)
        logits = model.logits(y_in, audio_features=audio_features)
        loss = F.cross_entropy(logits.transpose(1, 2), y_out)

        loss = loss / accum_grad_steps
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return total_loss


@torch.no_grad()
def evaluate(model: Whisper, dev_loader: DataLoader, dstore=None, args=None) -> float:
    model.eval()
    total_loss = 0

    for x, y_in, y_out in tqdm(dev_loader, desc="Evaluating"):
        x, y_in, y_out = (
            x.to(model.device),
            y_in.to(model.device),
            y_out.to(model.device),
        )

        # 1. Get audio features
        audio_features = model.embed_audio(x)

        # 2. Get raw hidden states from the decoder
        # The cross_attention block in the decoder gives us the raw states
        # Re-implementing decoder forward pass to get hidden states before final LayerNorm
        hidden_states = model.decoder.token_embedding(y_in) + model.decoder.positional_embedding[:y_in.shape[-1]]
        hidden_states = hidden_states.to(audio_features.dtype)
        for block in model.decoder.blocks:
            hidden_states = block(hidden_states, audio_features, mask=model.decoder.mask)

        # 3. Manually apply the final LayerNorm
        hidden_states_norm = model.decoder.ln(hidden_states)

        # 4. Manually calculate logits
        logits = hidden_states_norm.float() @ model.decoder.token_embedding.weight.t()

        if dstore is not None and args.use_knn:
            # We are predicting the next token, so we need to shift the labels and hidden states
            # IMPORTANT: Use the NORMALIZED hidden states for KNN query
            hidden_states_for_knn = hidden_states_norm[:, :-1, :]
            
            # THE BUG WAS HERE: The target labels were shifted by one position.
            # y_out is already the correct target for y_in.
            # When we slice the predictions to [:, :-1], we must slice the targets to match.
            y_out_shifted = y_out[:, :-1].clone()

            # Get kNN probabilities
            knn_probs = dstore.get_knn_probs(hidden_states_for_knn, k=args.knn_k)

            # Get model probabilities
            model_probs = F.softmax(logits[:, :-1, :], dim=-1)

            # Interpolate
            combined_probs = (
                1 - args.knn_lambda
            ) * model_probs + args.knn_lambda * knn_probs

            # Calculate loss using NLLLoss on the interpolated probabilities
            # Add a small epsilon to prevent log(0)
            log_probs = torch.log(combined_probs + 1e-9)
            log_probs_flat = log_probs.view(-1, log_probs.size(-1))
            y_out_flat = y_out_shifted.reshape(-1)
            loss = F.nll_loss(
                log_probs_flat, y_out_flat, ignore_index=-100
            )
        else:
            # Default cross-entropy loss
            loss = F.cross_entropy(logits.transpose(1, 2), y_out)

        total_loss += loss.item()
    return total_loss / len(dev_loader)


def build_dstore(args, model, tokenizer):
    try:
        import faiss
        import shutil
    except ImportError:
        raise ImportError(
            "Please install faiss-cpu or faiss-gpu: `pip install faiss-cpu`"
        )

    dstore_dir = Path(args.dstore_dir)
    dstore_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = dstore_dir / "temp_dstore"
    if temp_dir.exists():
        print(f"Temporary directory {temp_dir} already exists. Deleting it.")
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    fp16 = args.device == "cuda"
    datastore_loader = get_dataloader(
        json=args.train_json,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        fp16=fp16,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=model.dims.n_text_ctx // 2 - 1,
        prompt_use_rate=0.0,
        no_timestamps_rate=0.0,
        shuffle=False,
    )

    # --- Step 1: Feature Extraction (Chunk by Chunk) ---
    print("Step 1/4: Extracting features and saving to temporary files...")
    all_keys_list = []
    all_values_list = []
    chunk_idx = 0
    current_chunk_key_count = 0
    # Process and save features in chunks to avoid OOM
    # A chunk size of 500,000 keys (approx. 0.75 GB in memory per chunk for 'tiny' model)
    keys_per_chunk = 500_000

    for x, y_in, y_out in tqdm(datastore_loader, desc="Building datastore"):
        x, y_in, y_out = (
            x.to(model.device),
            y_in.to(model.device),
            y_out.to(model.device),
        )
        with torch.no_grad():
            audio_features = model.embed_audio(x)
            
            # Manually get normalized hidden states, consistent with the evaluate function
            hidden_states = model.decoder.token_embedding(y_in) + model.decoder.positional_embedding[:y_in.shape[-1]]
            hidden_states = hidden_states.to(audio_features.dtype)
            for block in model.decoder.blocks:
                hidden_states = block(hidden_states, audio_features, mask=model.decoder.mask)
            decoder_hidden_states = model.decoder.ln(hidden_states)

        keys = decoder_hidden_states.reshape(-1, decoder_hidden_states.size(-1))
        values = y_out.reshape(-1)

        non_pad_mask = values != -100
        keys = keys[non_pad_mask]
        values = values[non_pad_mask]

        if keys.size(0) > 0:
            all_keys_list.append(keys.cpu().numpy())
            all_values_list.append(values.cpu().numpy())
            current_chunk_key_count += keys.size(0)
        
        if current_chunk_key_count >= keys_per_chunk:
            keys_chunk = np.concatenate(all_keys_list, axis=0).astype("float32")
            values_chunk = np.concatenate(all_values_list, axis=0).astype("int64")
            np.save(temp_dir / f"keys_{chunk_idx}.npy", keys_chunk)
            np.save(temp_dir / f"values_{chunk_idx}.npy", values_chunk)
            tqdm.write(f"Saved chunk {chunk_idx} to disk with {current_chunk_key_count} keys.")
            chunk_idx += 1
            all_keys_list, all_values_list = [], []
            current_chunk_key_count = 0

    # Save the last remaining chunk
    if all_keys_list:
        keys_chunk = np.concatenate(all_keys_list, axis=0).astype("float32")
        values_chunk = np.concatenate(all_values_list, axis=0).astype("int64")
        np.save(temp_dir / f"keys_{chunk_idx}.npy", keys_chunk)
        np.save(temp_dir / f"values_{chunk_idx}.npy", values_chunk)
        tqdm.write(f"Saved final chunk {chunk_idx} to disk with {current_chunk_key_count} keys.")
        chunk_idx += 1
    
    key_files = sorted(list(temp_dir.glob("keys_*.npy")))
    if not key_files:
        print("No keys were generated for the datastore. Aborting.")
        return
        
    # --- Step 2: Train Faiss Index ---
    print("\nStep 2/4: Training faiss index...")
    # Load the first chunk to get dimensions and sample for training
    sample_keys = np.load(key_files[0])
    num_keys_total = sum(np.load(f, mmap_mode='r').shape[0] for f in key_files)
    d = sample_keys.shape[1]

    nlist = int(4 * np.sqrt(num_keys_total))
    nlist = min(max(nlist, 1024), 65536)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    
    # Train the index on a sample of the data for efficiency
    # For very large datasets, just training on the first chunk is often sufficient
    index.train(sample_keys)
    del sample_keys # free memory
    print("Index training complete.")

    # --- Step 3: Populate Faiss Index from Chunks ---
    print("\nStep 3/4: Adding keys to the index from temporary files...")
    for kf in tqdm(key_files, desc="Adding keys"):
        keys_to_add = np.load(kf)
        index.add(keys_to_add)

    faiss.write_index(index, str(dstore_dir / "dstore.index"))
    print("Index populated and saved.")

    # --- Step 4: Consolidate Values and Clean Up ---
    print("\nStep 4/4: Consolidating value files and cleaning up...")
    value_files = sorted(list(temp_dir.glob("values_*.npy")))
    all_values = np.concatenate([np.load(vf) for vf in tqdm(value_files, desc="Consolidating values")])
    np.save(dstore_dir / "dstore_values.npy", all_values)

    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)

    print(f"\nDatastore built with {index.ntotal} entries.")
    print(f"Datastore saved in {dstore_dir}")


class KNN_Dstore:
    def __init__(self, dstore_dir, vocab_size, device="cpu"):
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "Please install faiss-cpu or faiss-gpu: `pip install faiss-cpu`"
            )
        self.dstore_dir = dstore_dir
        self.index = faiss.read_index(str(Path(dstore_dir) / "dstore.index"))
        self.values = np.load(str(Path(dstore_dir) / "dstore_values.npy"))
        self.vocab_size = vocab_size
        self.device = device
        self.index.nprobe = 32 # for IVF indexes

    def get_knn_probs(self, queries, k):
        batch_size, seq_len, dim = queries.shape
        queries_flat = queries.reshape(-1, dim)
        queries_flat_np = queries_flat.detach().cpu().numpy().astype("float32")

        distances, indices = self.index.search(queries_flat_np, k)

        retrieved_values = torch.from_numpy(self.values[indices]).to(self.device)
        # (batch*seq_len, k)

        knn_probs = torch.zeros(
            queries_flat.shape[0], self.vocab_size, device=self.device
        )

        distances = torch.from_numpy(distances).to(self.device)
        weights = 1 / (distances + 1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        knn_probs.scatter_add_(1, retrieved_values, weights)

        return knn_probs.view(batch_size, seq_len, -1)


def save_model(model: Whisper, save_path: str) -> None:
    # save model in half precision to save space
    model = copy.deepcopy(model).half()
    # save model weights and config in a dictionary that can be loaded with `whisper.load_model`
    torch.save({"model_state_dict": model.state_dict(), "dims": asdict(model.dims)}, save_path)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_args(args: argparse.Namespace, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(vars(args), indent=4, ensure_ascii=False))


def infinite_iter(data_loader: DataLoader) -> Iterator:
    while True:
        for batch in data_loader:
            yield batch


def main_loop(
    model: Whisper,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args: argparse.Namespace,
    dstore=None,
) -> None:
    min_loss = evaluate(model, dev_loader, dstore, args)
    print(f"Initial loss: {min_loss}")
    pbar = tqdm(range(1, args.train_steps + 1))
    train_iter = infinite_iter(train_loader)
    for step in pbar:
        train_loss = train_step(
            model,
            train_iter,
            optimizer,
            scheduler,
            args.accum_grad_steps,
            args.train_only_decoder,
            args.max_grad_norm,
        )
        pbar.set_postfix({"loss": train_loss})

        if step % args.eval_steps == 0:
            eval_loss = evaluate(model, dev_loader, dstore, args)
            tqdm.write(f"Step {step}: validation loss={eval_loss}")
            if eval_loss < min_loss:
                min_loss = eval_loss
                save_model(model, f"{args.save_dir}/best_model.pt")

            if args.save_all_checkpoints:
                save_model(model, f"{args.save_dir}/step{step}.pt")

            save_model(model, f"{args.save_dir}/last_model.pt")


def main():
    args = get_parser().parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_args(args, f"{args.save_dir}/args.json")

    tokenizer = get_tokenizer(multilingual=".en" not in args.model, task="transcribe")
    model = whisper.load_model(args.model, args.device)

    if args.load_from_checkpoint:
        print(f"Loading model from {args.load_from_checkpoint}")
        checkpoint = torch.load(args.load_from_checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    if args.build_dstore:
        build_dstore(args, model, tokenizer)
        return

    # For standard training, we need both train and dev json.
    # For KNN evaluation or eval-only, we only need dev json.
    if not args.use_knn and not args.build_dstore and not args.eval_only:
        if args.train_json is None or args.dev_json is None:
            raise ValueError(
                "--train-json and --dev-json are required for standard training."
            )

    if args.dev_json is None and not args.build_dstore:
        raise ValueError("--dev-json is required for evaluation.")

    dstore = None
    if args.use_knn and not args.build_dstore:
        print("Loading KNN datastore...")
        dstore = KNN_Dstore(args.dstore_dir, model.dims.n_vocab, device=args.device)
        print("Datastore loaded.")

    #  -1 is for the special token `sot_prev` and the other half is for the transcribed tokens
    max_prompt_length = model.dims.n_text_ctx // 2 - 1

    fp16 = args.device == "cuda"
    train_loader = None
    # We only need a train_loader if we are actually training
    if not args.use_knn and not args.build_dstore and not args.eval_only and args.train_json:
        train_loader = get_dataloader(
            json=args.train_json,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            fp16=fp16,
            no_timestamps_training=args.no_timestamps_training,
            max_prompt_length=max_prompt_length,
            prompt_use_rate=args.prompt_use_rate,
            no_timestamps_rate=args.no_timestamps_rate,
            shuffle=True,
        )

    dev_loader = get_dataloader(
        json=args.dev_json,
        tokenizer=tokenizer,
        batch_size=args.dev_batch_size,
        fp16=fp16,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        # always use prompts and timestamps for validation to make it deterministic
        prompt_use_rate=1.0,
        no_timestamps_rate=0.0,
        shuffle=False,
    )
    if args.use_adam_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("For using Adam 8bit optimizer you need to have bitsandbytes installed.")
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_steps
    )

    # If we are just evaluating (with or without KNN), no training loop is needed.
    if args.eval_only or args.use_knn:
        if args.use_knn:
            print("Running kNN-augmented evaluation...")
        else:
            print("Running standard evaluation (without k-NN)...")
        
        eval_loss = evaluate(model, dev_loader, dstore, args)
        
        if args.use_knn:
            print(f"kNN-augmented validation loss: {eval_loss}")
        else:
            print(f"Standard validation loss: {eval_loss}")
        return

    main_loop(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        dstore=dstore,
    )


if __name__ == "__main__":
    main()
