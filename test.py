#!/usr/bin/env python3
"""
Document Conversion Benchmark Tool

Author: Mohanad Abu Nayla
GitHub: hannody
Year: 2025

A benchmarking tool for docling PDF conversion with GPU acceleration support.
"""

import json
import logging
import time
import psutil
from datetime import datetime
from pathlib import Path
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, RapidOcrOptions, TesseractOcrOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("GPUtil not available. Install with: uv add gputil")

# Set up logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def convert_pdf_document(source_path, warmup=False):
    """
    Convert PDF document using docling with GPU acceleration and accurate timing.
    """
    # Configure pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    # Configure EasyOCR specifically
    pipeline_options.ocr_options = EasyOcrOptions(
        force_full_page_ocr=True
    )

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=16,
        device=AcceleratorDevice.CUDA
    )

    # Initialize document converter
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Get GPU info before conversion
    gpu_info = None
    if HAS_GPUTIL and not warmup:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = {
                    'name': gpus[0].name,
                    'memory_total': gpus[0].memoryTotal,
                    'memory_used_before': gpus[0].memoryUsed
                }
        except Exception:
            pass

    # Convert document with precise timing
    if not warmup:
        print("Starting timed conversion...")
    start_time = time.perf_counter()  # More precise than time.time()
    conv_result = doc_converter.convert(source_path)
    end_time = time.perf_counter()
    processing_time = end_time - start_time

    # Get GPU memory after conversion
    if gpu_info and HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            gpu_info['memory_used_after'] = gpus[0].memoryUsed
            gpu_info['memory_peak'] = max(gpu_info['memory_used_before'], gpu_info['memory_used_after'])
        except Exception:
            pass

    if not warmup:
        _log.info(f"Document converted in {processing_time:.3f} seconds.")
        print(f"Document converted in {processing_time:.3f} seconds.")
        
        if gpu_info:
            print(f"GPU: {gpu_info['name']}")
            print(f"GPU Memory Used: {gpu_info['memory_used_after']}MB / {gpu_info['memory_total']}MB")

    return conv_result, processing_time, gpu_info


if __name__ == "__main__":
    source = "./9pages.pdf" # 17.x seconds on RTX 4070 Super.
    # source = "./chat_with_papers_image_based.pdf"

    # Multiple runs for better benchmarking
    num_runs = 3
    times = []
    
    try:
        print(f"Running {num_runs} benchmark iterations...")
        
        # Warm-up run (important for GPU benchmarking)
        print("\nPerforming warm-up run...")
        try:
            convert_pdf_document(source, warmup=True)
            print("Warm-up completed.")
        except Exception as e:
            print(f"Warm-up failed: {e}")
        
        # Benchmark runs
        gpu_info = None
        result = None
        
        for i in range(num_runs):
            print(f"\n--- Run {i+1}/{num_runs} ---")
            result, processing_time, gpu_info = convert_pdf_document(source)
            times.append(processing_time)
            
            if i < num_runs - 1:  # Don't sleep after last run
                print("Cooling down...")
                time.sleep(2)  # Cool down between runs
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n=== Benchmark Results ===")
        print(f"Runs: {num_runs}")
        print(f"Average: {avg_time:.3f}s")
        print(f"Min: {min_time:.3f}s")
        print(f"Max: {max_time:.3f}s")
        print(f"Individual times: {[f'{t:.3f}s' for t in times]}")
        
        print("Conversion completed successfully!")

        # Generate output filename with backend and timestamp
        backend_name = "easyocr"  # Change this based on OCR backend used
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        output_filename = f"output_{backend_name}_{timestamp}_avg{avg_time:.2f}s.md"

        # Write the result to the output file
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"# Document Conversion Benchmark\n\n")
            f.write(f"**Backend:** {backend_name}\n")
            f.write(f"**Runs:** {num_runs}\n")
            f.write(f"**Average Time:** {avg_time:.3f} seconds\n")
            f.write(f"**Min Time:** {min_time:.3f} seconds\n")
            f.write(f"**Max Time:** {max_time:.3f} seconds\n")
            f.write(f"**Individual Times:** {[f'{t:.3f}s' for t in times]}\n")
            f.write(f"**Timestamp:** {timestamp}\n")
            f.write(f"**Source File:** {source}\n")
            if gpu_info:
                f.write(f"**GPU:** {gpu_info['name']}\n")
                f.write(f"**GPU Memory:** {gpu_info['memory_used_after']}MB / {gpu_info['memory_total']}MB\n")
            f.write("\n## Document Content\n\n")
            f.write(result.document.export_to_markdown())

        print(f"\nBenchmark results written to: {output_filename}")

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
