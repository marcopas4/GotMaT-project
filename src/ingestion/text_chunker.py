import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import logging
from src.utils.logging_utils import setup_logger

class TextChunker:
    """Splits cleaned text into chunks for downstream processing."""

    def __init__(
        self,
        input_dir: str = "data/cleaned_text",
        output_dir: str = "data/chunked_text",
        max_chunk_words: int = 500,
        min_chunk_length: int = 10,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize TextChunker with configuration parameters.

        Args:
            input_dir (str): Directory containing cleaned text files.
            output_dir (str): Directory to save chunked text and metadata.
            max_chunk_words (int): Maximum words per chunk.
            min_chunk_length (int): Minimum character count for valid chunks.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_chunk_words = max_chunk_words
        self.min_chunk_length = min_chunk_length
        self.logger = logger or setup_logger("src.ingestion.text_chunker")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sentence boundary regex for Italian (handles periods, question marks, exclamation points)
        self.sentence_pattern = r'(?<=[.!?])\s+(?=[A-ZÀÈÌÒÙ])'

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex for Italian text.

        Args:
            text (str): Input text to split.

        Returns:
            List[str]: List of sentences.
        """
        if not text:
            return []
        try:
            sentences = re.split(self.sentence_pattern, text.strip())
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            self.logger.error("Sentence splitting failed: %s", str(e))
            return []

    def create_chunks(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Create chunks from sentences, respecting max_chunk_words.

        Args:
            sentences (List[str]): List of sentences.

        Returns:
            List[Dict[str, Any]]: List of chunks with metadata.
        """
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            word_count = len(sentence.split())
            if current_word_count + word_count <= self.max_chunk_words:
                current_chunk.append(sentence)
                current_word_count += word_count
            else:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "word_count": current_word_count,
                        "char_length": len(chunk_text),
                        "is_valid": self.is_valid_chunk(chunk_text)
                    })
                current_chunk = [sentence]
                current_word_count = word_count

        # Add the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "word_count": current_word_count,
                "char_length": len(chunk_text),
                "is_valid": self.is_valid_chunk(chunk_text)
            })

        return chunks

    def is_valid_chunk(self, text: str) -> bool:
        """
        Validate if a chunk is meaningful.

        Args:
            text (str): Chunk text to validate.

        Returns:
            bool: True if chunk is valid (sufficient length and content).
        """
        if not text or len(text.strip()) < self.min_chunk_length:
            return False

        # Check for Italian diacritics or legal terms
        if re.search(r'[àèìòù]', text, re.IGNORECASE) or re.search(r'\b(legge|decreto|articolo)\b', text, re.IGNORECASE):
            return True

        # Fallback: At least 3 words
        words = text.split()
        return len(words) >= 3

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single text file, chunk it, and save results.

        Args:
            file_path (Path): Path to the input text file.

        Returns:
            Dict[str, Any]: Chunking result with chunks and metadata.
        """
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "chunks": [],
            "is_valid": False,
            "error": None,
            "original_length": 0,
            "chunk_count": 0
        }

        self.logger.info("Chunking text file: %s", file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            result["original_length"] = len(text)

            # Split into sentences and create chunks
            sentences = self.split_into_sentences(text)
            chunks = self.create_chunks(sentences)
            result["chunks"] = chunks
            result["chunk_count"] = len(chunks)
            result["is_valid"] = any(chunk["is_valid"] for chunk in chunks)

            if not result["is_valid"]:
                result["error"] = "No valid chunks created (too short or lacks meaningful content)"
                self.logger.warning(result["error"])

            # Save each chunk (no per-file metadata any longer)
            self.save_chunks(file_path.name, chunks)
            return result

        except Exception as e:
            self.logger.error("Failed to process %s: %s", file_path, str(e))
            result["error"] = str(e)
            return result

    def save_chunks(self, file_name: str, chunks: List[Dict[str, Any]]) -> None:
        """
        Save each chunk's text to output directory, no metadata per chunk.

        Args:
            file_name (str): Name of original file.
            chunks (List[Dict[str, Any]]): Chunk dictionary list.
        """
        file_stem = file_name.rsplit(".", 1)[0]
        for i, chunk in enumerate(chunks, 1):
            chunk_file = self.output_dir / f"{file_stem}_chunk_{i:03d}.txt"
            try:
                with open(chunk_file, "w", encoding="utf-8") as f:
                    f.write(chunk["text"])
                self.logger.info("Saved chunk to %s", chunk_file)
            except Exception as e:
                self.logger.error("Failed to save chunk to %s: %s", chunk_file, str(e))

    def process_directory(self) -> None:
        """
        Process all text files in the input directory and save accumulated metadata once.
        """
        text_files = list(self.input_dir.glob("*.txt"))
        if not text_files:
            self.logger.warning("No text files found in %s", self.input_dir)
            return

        self.logger.info("Processing %d text files in %s", len(text_files), self.input_dir)
        processed_files = 0
        metadata_collection = []

        for file_path in text_files:
            result = self.process_file(file_path)
            # Record only the metadata, with chunk text excluded
            file_metadata = {
                "file_path": result["file_path"],
                "file_name": result["file_name"],
                "is_valid": result["is_valid"],
                "error": result["error"],
                "original_length": result["original_length"],
                "chunk_count": result["chunk_count"],
                "chunks_metadata": [
                    {
                        "chunk_id": f"{result['file_name'].rsplit('.', 1)[0]}_chunk_{i:03d}",
                        "word_count": chunk["word_count"],
                        "char_length": chunk["char_length"],
                        "is_valid": chunk["is_valid"]
                    }
                    for i, chunk in enumerate(result["chunks"], 1)
                ]
            }
            metadata_collection.append(file_metadata)
            processed_files += 1

        self.logger.info("Processed %d/%d text files", processed_files, len(text_files))

        # Save all metadata in a single summary JSON file
        summary_file = self.output_dir / "chunking_summary.json"
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(metadata_collection, f, ensure_ascii=False, indent=2)
            self.logger.info("Saved chunking summary to %s", summary_file)
        except Exception as e:
            self.logger.error("Failed to save chunking summary: %s", str(e))


if __name__ == "__main__":
    with open('src/configs/config.yaml') as file:
        config = yaml.safe_load(file)
    try:
        chunker = TextChunker(
            input_dir=config['cleaned_texts'].get('prefettura_v1.2', 'data/prefettura_v1.2_cleaned_texts'),
            output_dir=config['chunks'].get('prefettura_v1.2', 'data/chunks/prefettura_v1.2_chunks'),
            max_chunk_words=500,
            min_chunk_length=10
        )
        chunker.process_directory()
        print("Text chunking completed.")
    except Exception as e:
        print(f"Error during text chunking: {e}")
