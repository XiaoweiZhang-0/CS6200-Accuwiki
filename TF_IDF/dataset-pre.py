import gc
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cudf
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pynvml
import rmm
from nltk.corpus import stopwords
from numba import cuda
from tqdm import tqdm

# ✅ Configuration
input_folder = "/home/blackcatecho/.cache/huggingface/datasets/wikimedia___wikipedia/20231101.en/0.0.0/b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
output_folder = "wikipedia_processed"
batch_size = 10000  # Smaller batch size to reduce memory pressure
num_workers = max(1, cpu_count() - 1)  # Leave one CPU core free
max_files_per_run = 5  # Process this many files before restarting (to free memory)

# Create output directory
os.makedirs(output_folder, exist_ok=True)


def get_gpu_memory():
    """Return GPU total memory in bytes"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return info.total


def initialize_memory_pool():
    """Initialize RMM memory pool with more conservative values"""
    try:
        total_bytes = get_gpu_memory()
        # Use only 60% of available memory instead of 80%
        pool_size = int(total_bytes * 0.6)

        # Align to 256-byte boundary
        pool_size = (pool_size // 256) * 256

        print(
            f"🔧 Attempting to initialize RMM memory pool with {pool_size / 1e9:.2f} GB"
        )
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=pool_size,
            maximum_pool_size=pool_size,
        )
        print(f"✅ Memory pool initialization successful")
    except Exception as e:
        print(f"⚠️ Failed to initialize memory pool: {e}")
        print("🔧 Continuing with default allocator")


class GPUTextProcessor:
    def __init__(self):
        # Load stopwords once
        self.stopwords = set(stopwords.words("english"))

        # We'll use this to track if lemmatizer is initialized
        self.lemmatizer = None

    def _lazy_init_lemmatizer(self):
        """Lazily initialize the lemmatizer only when needed"""
        if self.lemmatizer is None:
            from nltk.stem import WordNetLemmatizer

            self.lemmatizer = WordNetLemmatizer()
        return self.lemmatizer

    def process_batch(self, df):
        """Process a batch of text data efficiently"""
        try:
            # Step 1: Text cleaning (keep on GPU)
            df["text"] = (
                df["text"]
                .str.normalize_spaces()  # Normalize whitespace
                .str.lower()  # Convert to lowercase
                .str.replace(r"[^\w\s]", " ")  # Replace punctuation with space
                .str.normalize_spaces()  # Clean up again after replacement
            )

            # Step 2: GPU tokenization - much faster than CPU
            df["tokens"] = df["text"].str.split()

            # Step 3: Convert tokens to pandas for CPU operations
            tokens_pd = df["tokens"].to_pandas()

            # Step 4: Filter stopwords and lemmatize in one pass
            # This combines two operations to reduce CPU-GPU transfers
            lemmatizer = self._lazy_init_lemmatizer()

            def process_token_list(token_list):
                return [
                    lemmatizer.lemmatize(token)
                    for token in token_list
                    if token not in self.stopwords and len(token) > 1
                ]

            # Process all rows at once
            processed_tokens = tokens_pd.apply(process_token_list)

            # Step 5: Update the dataframe
            df["tokens"] = cudf.Series.from_pandas(processed_tokens)

            return df

        except Exception as e:
            print(f"❌ Error processing batch: {str(e)}")
            raise e


def process_file(file_info):
    """Process a single Arrow file"""
    input_path, output_path = file_info

    # Initialize RMM for this file
    initialize_memory_pool()

    # Create processor for this file
    processor = GPUTextProcessor()

    try:
        # Create output schema with tokens column
        output_fields = None
        first_batch_processed = False

        # Read and process the file in batches
        with pa.memory_map(input_path, "rb") as source:
            reader = pa.ipc.open_stream(source)

            # Process in batches with progress bar
            for batch_idx, batch in enumerate(
                tqdm(reader, desc=f"Processing {Path(input_path).name}")
            ):
                # Convert to Table then to cuDF DataFrame
                table = pa.Table.from_batches([batch])
                df = cudf.DataFrame.from_arrow(table)

                # Process the batch
                processed_df = processor.process_batch(df)

                # Convert back to Arrow
                processed_table = processed_df.to_arrow()

                # For the first batch, get schema and create writer
                if not first_batch_processed:
                    output_fields = processed_table.schema
                    writer = pa.parquet.ParquetWriter(output_path, output_fields)
                    first_batch_processed = True

                # Write batch
                writer.write_table(processed_table)

                # Cleanup to free memory
                del df, processed_df, processed_table, table
                gc.collect()

        # Close writer after all batches
        if first_batch_processed:
            writer.close()

        print(f"✅ Successfully processed {Path(input_path).name}")
        return True

    except Exception as e:
        print(f"❌ Error processing file {input_path}: {str(e)}")
        return False


def main():
    # Get all Arrow files
    arrow_files = [f for f in os.listdir(input_folder) if f.endswith(".arrow")]

    if not arrow_files:
        print(f"❌ No Arrow files found in {input_folder}")
        return

    print(f"🔍 Found {len(arrow_files)} Arrow files to process")

    # Create file processing tasks
    tasks = []
    for arrow_file in arrow_files:
        input_path = os.path.join(input_folder, arrow_file)
        output_path = os.path.join(output_folder, f"{Path(arrow_file).stem}.parquet")
        tasks.append((input_path, output_path))

    # Process in groups to manage memory
    for i in range(0, len(tasks), max_files_per_run):
        batch_tasks = tasks[i : i + max_files_per_run]

        print(
            f"🚀 Processing batch {i//max_files_per_run + 1}/{(len(tasks)-1)//max_files_per_run + 1}"
        )

        # Process files in the batch sequentially
        # (parallel processing with multiple GPUs would require more complex code)
        for task in batch_tasks:
            process_file(task)

    print(f"✅ All files processed. Results saved to {output_folder}")


if __name__ == "__main__":
    main()
