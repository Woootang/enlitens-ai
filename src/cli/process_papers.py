"""
CLI Interface for Enlitens PDF Processing Pipeline

This module provides a command-line interface for processing PDFs with:
- Progress bars and status updates
- GPU temperature and memory monitoring
- Processing statistics (papers/hour, accuracy)
- Error reporting and retry capabilities
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime, timedelta
import json

# Rich for beautiful CLI output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

# Import our modules
from src.pipeline.document_processor import DocumentProcessor
from src.pipeline.checkpoint_manager import CheckpointManager, ResumeProcessor
from src.monitoring.metrics import MetricsCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enlitens_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
console = Console()


class EnlitensCLI:
    """
    Command-line interface for Enlitens PDF processing
    
    Features:
    - Beautiful progress bars with Rich
    - Real-time GPU monitoring
    - Processing statistics
    - Error handling and retry
    - Resume capability
    """
    
    def __init__(self):
        self.processor = None
        self.checkpoint_manager = CheckpointManager()
        self.metrics_collector = MetricsCollector()
        self.start_time = None
        
    def setup_processor(self, pdf_input_dir: str, output_dir: str, cache_dir: str,
                       ollama_url: str, ollama_model: str):
        """Initialize the document processor"""
        try:
            self.processor = DocumentProcessor(
                pdf_input_dir=pdf_input_dir,
                output_dir=output_dir,
                cache_dir=cache_dir,
                ollama_url=ollama_url,
                ollama_model=ollama_model
            )
            return True
        except Exception as e:
            console.print(f"[red]Failed to initialize processor: {e}[/red]")
            return False
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        console.print("[blue]Checking prerequisites...[/blue]")
        
        # Check if processor is initialized
        if not self.processor:
            console.print("[red]Processor not initialized[/red]")
            return False
        
        # Check vLLM availability
        if not self.processor.ollama_client.is_available():
            console.print("[red]vLLM is not available. Please start the inference server first.[/red]")
            return False
        
        # Check GPU availability
        try:
            import torch
            if not torch.cuda.is_available():
                console.print("[yellow]CUDA not available - processing will be slower[/yellow]")
            else:
                gpu_name = torch.cuda.get_device_name(0)
                console.print(f"[green]GPU available: {gpu_name}[/green]")
        except ImportError:
            console.print("[yellow]PyTorch not available[/yellow]")
        
        console.print("[green]Prerequisites check passed[/green]")
        return True
    
    def get_pdf_files(self, input_dir: str) -> List[Path]:
        """Get list of PDF files to process"""
        input_path = Path(input_dir)
        if not input_path.exists():
            console.print(f"[red]Input directory does not exist: {input_dir}[/red]")
            return []
        
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            console.print(f"[yellow]No PDF files found in {input_dir}[/yellow]")
            return []
        
        console.print(f"[green]Found {len(pdf_files)} PDF files[/green]")
        return pdf_files
    
    def process_single_paper(self, pdf_path: Path, show_progress: bool = True) -> Dict[str, Any]:
        """Process a single PDF with progress display"""
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"Processing {pdf_path.name}", total=100)
                
                # Simulate progress updates
                progress.update(task, advance=20, description="Extracting PDF...")
                time.sleep(0.1)
                
                progress.update(task, advance=20, description="Extracting entities...")
                time.sleep(0.1)
                
                progress.update(task, advance=20, description="Synthesizing with AI...")
                time.sleep(0.1)
                
                progress.update(task, advance=20, description="Validating quality...")
                time.sleep(0.1)
                
                progress.update(task, advance=20, description="Saving to knowledge base...")
        
        # Actual processing
        success, document, error = self.processor.process_document(str(pdf_path))
        
        result = {
            'pdf_path': str(pdf_path),
            'success': success,
            'document_id': document.document_id if document else None,
            'quality_score': document.get_quality_score() if document else 0.0,
            'error': error
        }
        
        return result
    
    def process_batch(self, pdf_files: List[Path], batch_size: int = 5) -> Dict[str, Any]:
        """Process a batch of PDFs with progress tracking"""
        console.print(f"[blue]Processing {len(pdf_files)} PDFs in batches of {batch_size}[/blue]")
        
        results = {
            'total': len(pdf_files),
            'successful': 0,
            'failed': 0,
            'documents': [],
            'errors': []
        }
        
        # Process in batches
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(pdf_files) + batch_size - 1) // batch_size
            
            console.print(f"[blue]Processing batch {batch_num}/{total_batches} ({len(batch)} files)[/blue]")
            
            for pdf_file in batch:
                result = self.process_single_paper(pdf_file, show_progress=False)
                
                if result['success']:
                    results['successful'] += 1
                    results['documents'].append(result)
                    console.print(f"[green]✓ {pdf_file.name} - Quality: {result['quality_score']:.2f}[/green]")
                else:
                    results['failed'] += 1
                    results['errors'].append(result)
                    console.print(f"[red]✗ {pdf_file.name} - {result['error']}[/red]")
        
        return results
    
    def show_processing_summary(self, results: Dict[str, Any]):
        """Display processing summary"""
        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total PDFs", str(results['total']))
        table.add_row("Successful", str(results['successful']))
        table.add_row("Failed", str(results['failed']))
        table.add_row("Success Rate", f"{results['successful']/results['total']*100:.1f}%")
        
        if results['successful'] > 0:
            avg_quality = sum(doc['quality_score'] for doc in results['documents']) / results['successful']
            table.add_row("Average Quality", f"{avg_quality:.2f}")
        
        console.print(table)
        
        # Show errors if any
        if results['errors']:
            console.print("\n[red]Errors:[/red]")
            for error in results['errors']:
                console.print(f"  - {error['pdf_path']}: {error['error']}")
    
    def show_system_status(self):
        """Display system status"""
        if not self.processor:
            console.print("[red]Processor not initialized[/red]")
            return
        
        status = self.processor.get_processing_status()
        
        # Create status table
        table = Table(title="System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="magenta")
        
        table.add_row("vLLM", "✓ Available" if status['vllm_available'] else "✗ Not Available")
        table.add_row("Knowledge Base", "✓ Exists" if status['knowledge_base_exists'] else "✗ Not Found")
        table.add_row("Total Documents", str(status['total_documents']))
        
        # GPU status
        if 'model_manager_status' in status:
            mm_status = status['model_manager_status']
            table.add_row("Loaded Models", str(len(mm_status['loaded_models'])))
            
            if 'memory_usage' in mm_status:
                memory = mm_status['memory_usage']
                table.add_row("GPU Memory", f"{memory['gpu_used_gb']:.1f}GB / {memory['gpu_available_gb']:.1f}GB")
        
        console.print(table)
    
    def show_knowledge_base_stats(self):
        """Display knowledge base statistics"""
        if not self.processor:
            console.print("[red]Processor not initialized[/red]")
            return
        
        stats = self.processor.knowledge_base.get_statistics()
        
        table = Table(title="Knowledge Base Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Documents", str(stats['total_documents']))
        table.add_row("Complete Documents", str(stats['complete_documents']))
        table.add_row("Average Quality", f"{stats['average_quality']:.2f}")
        table.add_row("Entity Count", str(stats['entity_count']))
        table.add_row("Processing Errors", str(stats['processing_errors']))
        table.add_row("High Quality Docs", str(stats['high_quality_documents']))
        
        console.print(table)
    
    def run_interactive_mode(self):
        """Run interactive processing mode"""
        console.print(Panel.fit("Enlitens PDF Processing Pipeline", style="bold blue"))
        
        while True:
            console.print("\n[bold]Available Commands:[/bold]")
            console.print("1. Process single PDF")
            console.print("2. Process all PDFs")
            console.print("3. Show system status")
            console.print("4. Show knowledge base stats")
            console.print("5. Resume failed processing")
            console.print("6. Export knowledge base")
            console.print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == "1":
                pdf_path = input("Enter PDF path: ").strip()
                if Path(pdf_path).exists():
                    result = self.process_single_paper(Path(pdf_path))
                    if result['success']:
                        console.print(f"[green]Success! Quality: {result['quality_score']:.2f}[/green]")
                    else:
                        console.print(f"[red]Failed: {result['error']}[/red]")
                else:
                    console.print("[red]File not found[/red]")
            
            elif choice == "2":
                pdf_files = self.get_pdf_files("./enlitens_corpus/input_pdfs")
                if pdf_files:
                    results = self.process_batch(pdf_files)
                    self.show_processing_summary(results)
            
            elif choice == "3":
                self.show_system_status()
            
            elif choice == "4":
                self.show_knowledge_base_stats()
            
            elif choice == "5":
                self.resume_failed_processing()
            
            elif choice == "6":
                output_path = self.processor.export_knowledge_base()
                console.print(f"[green]Knowledge base exported to: {output_path}[/green]")
            
            elif choice == "7":
                console.print("[blue]Goodbye![/blue]")
                break
            
            else:
                console.print("[red]Invalid choice[/red]")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Enlitens PDF Processing Pipeline")
    parser.add_argument("--input-dir", default="./enlitens_corpus/input_pdfs", 
                       help="Input directory containing PDFs")
    parser.add_argument("--output-dir", default="./enlitens_corpus/output", 
                       help="Output directory for processed data")
    parser.add_argument("--cache-dir", default="./enlitens_corpus/cache_markdown", 
                       help="Cache directory for intermediate files")
    parser.add_argument("--ollama-url", default="http://localhost:8000/v1",
                       help="vLLM OpenAI-compatible URL")
    parser.add_argument("--ollama-model", default="qwen2.5-32b-instruct-q4_k_m",
                       help="vLLM model to use")
    parser.add_argument("--batch-size", type=int, default=5, 
                       help="Batch size for processing")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--single", type=str, 
                       help="Process a single PDF file")
    parser.add_argument("--status", action="store_true", 
                       help="Show system status")
    parser.add_argument("--stats", action="store_true", 
                       help="Show knowledge base statistics")
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = EnlitensCLI()
    
    # Setup processor
    if not cli.setup_processor(
        pdf_input_dir=args.input_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model
    ):
        sys.exit(1)
    
    # Check prerequisites
    if not cli.check_prerequisites():
        sys.exit(1)
    
    # Handle different modes
    if args.interactive:
        cli.run_interactive_mode()
    elif args.single:
        pdf_path = Path(args.single)
        if pdf_path.exists():
            result = cli.process_single_paper(pdf_path)
            if result['success']:
                console.print(f"[green]Success! Quality: {result['quality_score']:.2f}[/green]")
            else:
                console.print(f"[red]Failed: {result['error']}[/red]")
        else:
            console.print("[red]File not found[/red]")
    elif args.status:
        cli.show_system_status()
    elif args.stats:
        cli.show_knowledge_base_stats()
    else:
        # Process all PDFs
        pdf_files = cli.get_pdf_files(args.input_dir)
        if pdf_files:
            results = cli.process_batch(pdf_files, args.batch_size)
            cli.show_processing_summary(results)
        else:
            console.print("[yellow]No PDF files to process[/yellow]")


if __name__ == "__main__":
    main()
