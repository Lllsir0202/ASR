import argparse
import numpy as np
import faiss
from pathlib import Path
import shutil
from tqdm import tqdm
import os

def main():
    parser = argparse.ArgumentParser(description="Builds a Faiss index by creating and merging individual chunk indexes.")
    parser.add_argument(
        "--dstore_dir",
        type=str,
        required=True,
        help="Directory where the final datastore will be saved and temp files are located.",
    )
    args = parser.parse_args()

    dstore_dir = Path(args.dstore_dir)
    temp_dir = dstore_dir / "temp_dstore"
    chunk_index_dir = dstore_dir / "chunk_indexes"
    trained_index_skeleton_file = dstore_dir / "trained.index" 

    if not temp_dir.exists():
        print(f"Error: Temporary directory {temp_dir} not found. Please run the feature extraction step first.")
        return

    key_files = sorted(list(temp_dir.glob("keys_*.npy")))
    if not key_files:
        print("No temporary key files found. Aborting.")
        return
    
    chunk_index_dir.mkdir(exist_ok=True)

    if not trained_index_skeleton_file.exists():
        print("Fatal Error: trained.index not found. This file is crucial and should have been created. Aborting.")
        return
    
    index_skeleton = faiss.read_index(str(trained_index_skeleton_file))

    print("\n--- Phase 1: Creating index for each data chunk ---")
    num_chunks = len(key_files)
    for i in range(num_chunks):
        chunk_key_file = temp_dir / f"keys_{i}.npy"
        chunk_index_file = chunk_index_dir / f"chunk_{i}.index"

        if chunk_index_file.exists():
            print(f"Index for chunk {i} already exists. Skipping.")
            continue
        
        print(f"Building index for chunk {i}...")
        keys_to_add = np.load(chunk_key_file)
        
        index_chunk = faiss.clone_index(index_skeleton)
        # We must use direct_map to add with non-contiguous IDs later
        index_chunk.add(keys_to_add) # Simpler add, IDs will be handled by merge

        print(f"  - Added {index_chunk.ntotal} keys.")
        faiss.write_index(index_chunk, str(chunk_index_file))
        print(f"  - Saved index for chunk {i} to {chunk_index_file}")
        del index_chunk, keys_to_add

    print("\n--- Phase 2: Merging all chunk indexes into one final index ---")
    final_index_file = dstore_dir / "dstore.index"
    
    chunk_files = sorted(list(chunk_index_dir.glob("chunk_*.index")))
    
    # We use the skeleton as the base for the merged index
    merged_index = faiss.read_index(str(trained_index_skeleton_file))

    print("Merging indexes using `merge_from`...")
    for i in tqdm(range(len(chunk_files)), desc="Merging indexes"):
        index_to_merge = faiss.read_index(str(chunk_files[i]))
        merged_index.merge_from(index_to_merge, index_to_merge.ntotal)
        del index_to_merge
    
    print("\nMerge complete. Saving final index...")
    faiss.write_index(merged_index, str(final_index_file))
    print(f"Final index with {merged_index.ntotal} keys saved to {final_index_file}")

    print("\n--- Phase 3: Finalizing datastore and cleaning up ---")
    value_files = sorted(list(temp_dir.glob("values_*.npy")))
    all_values = np.concatenate([np.load(vf) for vf in tqdm(value_files, desc="Consolidating values")])
    np.save(dstore_dir / "dstore_values.npy", all_values)

    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    shutil.rmtree(chunk_index_dir)
    if trained_index_skeleton_file.exists():
        os.remove(trained_index_skeleton_file)

    print(f"\nDatastore built successfully with {merged_index.ntotal} entries.")

if __name__ == "__main__":
    main()