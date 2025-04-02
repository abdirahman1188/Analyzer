import os
import fitz  # PyMuPDF
import logging
from typing import List, Dict, Any
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
            
        # Basic cleaning
        text = text.replace('\n\n', ' [PARA] ')  # Mark paragraphs
        text = ' '.join(text.split())  # Remove excess whitespace
        text = text.replace(' [PARA] ', '\n\n')  # Restore paragraphs
        
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def chunk_text(text: str, filename: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks of specified size with overlap.
    
    Args:
        text: Text to chunk
        filename: Source filename for logging
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        logging.warning(f"No text to chunk for {filename}")
        return []
    
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk size, save current chunk and start new one
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            # Start new chunk with overlap by including last part of previous chunk
            current_chunk = get_overlap_text(current_chunk, chunk_overlap) + paragraph
        else:
            # Add separator if not first paragraph in chunk
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    logging.info(f"Created {len(chunks)} chunks from {filename}")
    return chunks

def get_overlap_text(text: str, overlap_size: int) -> str:
    """
    Get the last portion of text for chunk overlap.
    
    Args:
        text: Source text
        overlap_size: Size of overlap in characters
        
    Returns:
        Text for overlap
    """
    # If text is shorter than overlap size, return the whole text
    if len(text) <= overlap_size:
        return text
    
    # Get last 'overlap_size' characters
    overlap_text = text[-overlap_size:]
    
    # Try to start at a sentence or paragraph boundary for better context
    for separator in ['\n\n', '. ', '! ', '? ']:
        sep_pos = overlap_text.find(separator)
        if sep_pos >= 0:
            return overlap_text[sep_pos + len(separator):]
    
    return overlap_text

def process_pdfs(papers_dir: str, output_dir: str = None) -> List[Dict[str, Any]]:
    """
    Process all PDF documents in a directory.
    
    Args:
        papers_dir: Directory containing PDF papers
        output_dir: Directory to save processed chunks (optional)
        
    Returns:
        List of documents with metadata and chunks
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(papers_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logging.warning(f"No PDF files found in {papers_dir}")
        return []
    
    logging.info(f"Found {len(pdf_files)} PDF files in {papers_dir}")
    
    processed_docs = []
    
    # Process each PDF file
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(papers_dir, pdf_file)
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        if not text:
            logging.warning(f"No text extracted from {pdf_file}, skipping")
            continue
        
        # Create chunks
        chunks = chunk_text(text, pdf_file)
        
        # Save chunks if output directory is provided
        if output_dir and chunks:
            output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}_chunks.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(chunks))
            
            logging.info(f"Saved chunks to {output_file}")
        
        # Add to processed documents
        processed_docs.append({
            "filename": pdf_file,
            "path": pdf_path,
            "chunk_count": len(chunks),
            "chunks": chunks
        })
    
    logging.info(f"Processed {len(processed_docs)} documents")
    return processed_docs

if __name__ == "__main__":
    # Simple test run when script is executed directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDF documents for AI readiness analysis")
    parser.add_argument("--input_dir", default="data/papers", help="Directory containing PDF papers")
    parser.add_argument("--output_dir", default="data/chunks", help="Directory to save processed chunks")
    
    args = parser.parse_args()
    
    # Process PDFs
    process_pdfs(args.input_dir, args.output_dir)