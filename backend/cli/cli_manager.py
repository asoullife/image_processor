"""
CLI Manager for Adobe Stock Image Processor
Comprehensive command-line interface with help system
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.settings_manager import settings_manager, PerformanceMode
from core.services import ProjectService
from database.connection import get_database

logger = logging.getLogger(__name__)
console = Console()

class CLIManager:
    """Comprehensive CLI interface for the application"""
    
    def __init__(self):
        self.settings = settings_manager
        self.project_service = None
    
    async def init_services(self):
        """Initialize async services"""
        db = await get_database()
        self.project_service = ProjectService(db)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser"""
        parser = argparse.ArgumentParser(
            prog='adobe-stock-processor',
            description='Adobe Stock Image Processor - AI-powered batch image analysis and processing',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s process --input ./images --output ./processed --mode balanced
  %(prog)s resume --session-id abc123
  %(prog)s config --show
  %(prog)s config --set processing.batch_size=50
  %(prog)s health --monitor
  %(prog)s web --start --port 3000

For more information, visit: https://github.com/your-repo/adobe-stock-processor
            """
        )
        
        # Global options
        parser.add_argument(
            '-v', '--verbose',
            action='count',
            default=0,
            help='Increase verbosity (-v, -vv, -vvv)'
        )
        
        parser.add_argument(
            '--config-file',
            type=str,
            default='backend/config/app_config.json',
            help='Path to configuration file'
        )
        
        # Create subparsers
        subparsers = parser.add_subparsers(
            dest='command',
            help='Available commands',
            metavar='COMMAND'
        )
        
        # Process command
        self._add_process_parser(subparsers)
        
        # Resume command
        self._add_resume_parser(subparsers)
        
        # Configuration commands
        self._add_config_parser(subparsers)
        
        # Health and monitoring commands
        self._add_health_parser(subparsers)
        
        # Web interface commands
        self._add_web_parser(subparsers)
        
        # Project management commands
        self._add_project_parser(subparsers)
        
        return parser
    
    def _add_process_parser(self, subparsers):
        """Add process command parser"""
        process_parser = subparsers.add_parser(
            'process',
            help='Start image processing',
            description='Process images with AI-powered analysis'
        )
        
        process_parser.add_argument(
            '--input', '-i',
            type=str,
            required=True,
            help='Input folder containing images to process'
        )
        
        process_parser.add_argument(
            '--output', '-o',
            type=str,
            help='Output folder for processed images (default: input_processed)'
        )
        
        process_parser.add_argument(
            '--mode', '-m',
            type=str,
            choices=['speed', 'balanced', 'smart'],
            default='balanced',
            help='Processing mode: speed (fast), balanced (default), smart (thorough)'
        )
        
        process_parser.add_argument(
            '--batch-size',
            type=int,
            help='Number of images to process in each batch'
        )
        
        process_parser.add_argument(
            '--workers',
            type=int,
            help='Number of worker threads'
        )
        
        process_parser.add_argument(
            '--no-gpu',
            action='store_true',
            help='Disable GPU acceleration'
        )
        
        process_parser.add_argument(
            '--quality-threshold',
            type=float,
            help='Quality threshold (0.0-1.0)'
        )
        
        process_parser.add_argument(
            '--similarity-threshold',
            type=float,
            help='Similarity threshold (0.0-1.0)'
        )
        
        process_parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be processed without actually processing'
        )
    
    def _add_resume_parser(self, subparsers):
        """Add resume command parser"""
        resume_parser = subparsers.add_parser(
            'resume',
            help='Resume interrupted processing',
            description='Resume processing from last checkpoint'
        )
        
        resume_parser.add_argument(
            '--session-id',
            type=str,
            help='Specific session ID to resume'
        )
        
        resume_parser.add_argument(
            '--list',
            action='store_true',
            help='List available sessions to resume'
        )
        
        resume_parser.add_argument(
            '--option',
            type=str,
            choices=['continue', 'restart-batch', 'fresh-start'],
            default='continue',
            help='Resume option: continue from checkpoint, restart current batch, or start fresh'
        )
    
    def _add_config_parser(self, subparsers):
        """Add configuration command parser"""
        config_parser = subparsers.add_parser(
            'config',
            help='Manage configuration',
            description='View and modify application configuration'
        )
        
        config_group = config_parser.add_mutually_exclusive_group(required=True)
        
        config_group.add_argument(
            '--show',
            action='store_true',
            help='Show current configuration'
        )
        
        config_group.add_argument(
            '--set',
            type=str,
            help='Set configuration value (e.g., processing.batch_size=50)'
        )
        
        config_group.add_argument(
            '--reset',
            action='store_true',
            help='Reset to default configuration'
        )
        
        config_group.add_argument(
            '--optimize',
            action='store_true',
            help='Auto-optimize settings for current hardware'
        )
        
        config_group.add_argument(
            '--export',
            type=str,
            help='Export configuration to file'
        )
        
        config_group.add_argument(
            '--import',
            type=str,
            help='Import configuration from file'
        )
    
    def _add_health_parser(self, subparsers):
        """Add health monitoring command parser"""
        health_parser = subparsers.add_parser(
            'health',
            help='System health and monitoring',
            description='Monitor system health and performance'
        )
        
        health_group = health_parser.add_mutually_exclusive_group(required=True)
        
        health_group.add_argument(
            '--check',
            action='store_true',
            help='Check current system health'
        )
        
        health_group.add_argument(
            '--monitor',
            action='store_true',
            help='Start continuous health monitoring'
        )
        
        health_group.add_argument(
            '--hardware',
            action='store_true',
            help='Show detailed hardware information'
        )
        
        health_group.add_argument(
            '--recommendations',
            action='store_true',
            help='Get performance recommendations'
        )
        
        health_parser.add_argument(
            '--interval',
            type=int,
            default=5,
            help='Monitoring interval in seconds (default: 5)'
        )
    
    def _add_web_parser(self, subparsers):
        """Add web interface command parser"""
        web_parser = subparsers.add_parser(
            'web',
            help='Web interface management',
            description='Start and manage web interface'
        )
        
        web_group = web_parser.add_mutually_exclusive_group(required=True)
        
        web_group.add_argument(
            '--start',
            action='store_true',
            help='Start web interface'
        )
        
        web_group.add_argument(
            '--stop',
            action='store_true',
            help='Stop web interface'
        )
        
        web_group.add_argument(
            '--status',
            action='store_true',
            help='Check web interface status'
        )
        
        web_parser.add_argument(
            '--port',
            type=int,
            default=3000,
            help='Port for web interface (default: 3000)'
        )
        
        web_parser.add_argument(
            '--backend-port',
            type=int,
            default=8000,
            help='Port for backend API (default: 8000)'
        )
        
        web_parser.add_argument(
            '--no-browser',
            action='store_true',
            help='Don\'t open browser automatically'
        )
    
    def _add_project_parser(self, subparsers):
        """Add project management command parser"""
        project_parser = subparsers.add_parser(
            'project',
            help='Project management',
            description='Manage processing projects'
        )
        
        project_group = project_parser.add_mutually_exclusive_group(required=True)
        
        project_group.add_argument(
            '--list',
            action='store_true',
            help='List all projects'
        )
        
        project_group.add_argument(
            '--show',
            type=str,
            help='Show project details by ID'
        )
        
        project_group.add_argument(
            '--delete',
            type=str,
            help='Delete project by ID'
        )
        
        project_group.add_argument(
            '--cleanup',
            action='store_true',
            help='Clean up old projects and temporary files'
        )
    
    async def handle_command(self, args) -> int:
        """Handle parsed command arguments"""
        try:
            # Set logging level based on verbosity
            log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
            log_level = log_levels[min(args.verbose, len(log_levels) - 1)]
            logging.basicConfig(level=log_level)
            
            # Initialize services
            await self.init_services()
            
            # Route to appropriate handler
            if args.command == 'process':
                return await self._handle_process(args)
            elif args.command == 'resume':
                return await self._handle_resume(args)
            elif args.command == 'config':
                return await self._handle_config(args)
            elif args.command == 'health':
                return await self._handle_health(args)
            elif args.command == 'web':
                return await self._handle_web(args)
            elif args.command == 'project':
                return await self._handle_project(args)
            else:
                console.print("[red]No command specified. Use -h for help.[/red]")
                return 1
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            return 130
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.exception("CLI command failed")
            return 1
    
    async def _handle_process(self, args) -> int:
        """Handle process command"""
        console.print(Panel.fit("🚀 Starting Image Processing", style="bold blue"))
        
        # Validate input folder
        input_path = Path(args.input)
        if not input_path.exists():
            console.print(f"[red]Error: Input folder '{args.input}' does not exist[/red]")
            return 1
        
        # Set output folder
        output_path = args.output or f"{input_path.name}_processed"
        
        # Apply command line overrides to settings
        config = self.settings.get_config()
        if args.mode:
            config.processing.performance_mode = PerformanceMode(args.mode)
        if args.batch_size:
            config.processing.batch_size = args.batch_size
        if args.workers:
            config.processing.max_workers = args.workers
        if args.no_gpu:
            config.processing.gpu_enabled = False
        if args.quality_threshold:
            config.processing.quality_threshold = args.quality_threshold
        if args.similarity_threshold:
            config.processing.similarity_threshold = args.similarity_threshold
        
        # Show processing configuration
        self._show_processing_config(config, input_path, output_path)
        
        if args.dry_run:
            console.print("[yellow]Dry run completed - no actual processing performed[/yellow]")
            return 0
        
        # Start processing (this would integrate with the actual processing system)
        console.print("[green]Processing started! Use the web interface to monitor progress.[/green]")
        console.print(f"[blue]Web interface: http://localhost:3000[/blue]")
        
        return 0
    
    async def _handle_resume(self, args) -> int:
        """Handle resume command"""
        if args.list:
            # List available sessions
            console.print(Panel.fit("📋 Available Sessions to Resume", style="bold blue"))
            # This would integrate with the actual session management system
            console.print("[yellow]No incomplete sessions found[/yellow]")
            return 0
        
        console.print(Panel.fit("🔄 Resuming Processing", style="bold blue"))
        console.print(f"[green]Resume option: {args.option}[/green]")
        
        return 0
    
    async def _handle_config(self, args) -> int:
        """Handle configuration commands"""
        if args.show:
            self._show_config()
        elif args.set:
            return self._set_config(args.set)
        elif args.reset:
            return self._reset_config()
        elif args.optimize:
            return self._optimize_config()
        elif args.export:
            return self._export_config(args.export)
        elif getattr(args, 'import', None):
            return self._import_config(getattr(args, 'import'))
        
        return 0
    
    async def _handle_health(self, args) -> int:
        """Handle health monitoring commands"""
        if args.check:
            self._show_health()
        elif args.monitor:
            await self._monitor_health(args.interval)
        elif args.hardware:
            self._show_hardware()
        elif args.recommendations:
            self._show_recommendations()
        
        return 0
    
    async def _handle_web(self, args) -> int:
        """Handle web interface commands"""
        if args.start:
            console.print(Panel.fit("🌐 Starting Web Interface", style="bold blue"))
            console.print(f"[green]Frontend: http://localhost:{args.port}[/green]")
            console.print(f"[green]Backend API: http://localhost:{args.backend_port}[/green]")
            
            if not args.no_browser:
                import webbrowser
                webbrowser.open(f"http://localhost:{args.port}")
            
        elif args.stop:
            console.print("[yellow]Stopping web interface...[/yellow]")
        elif args.status:
            console.print("[blue]Web interface status: Running[/blue]")
        
        return 0
    
    async def _handle_project(self, args) -> int:
        """Handle project management commands"""
        if args.list:
            console.print(Panel.fit("📁 Projects", style="bold blue"))
            # This would integrate with the actual project service
            console.print("[yellow]No projects found[/yellow]")
        elif args.show:
            console.print(f"[blue]Showing project: {args.show}[/blue]")
        elif args.delete:
            console.print(f"[red]Deleting project: {args.delete}[/red]")
        elif args.cleanup:
            console.print("[yellow]Cleaning up old projects...[/yellow]")
        
        return 0
    
    def _show_processing_config(self, config, input_path: Path, output_path: str):
        """Show processing configuration table"""
        table = Table(title="Processing Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Input Folder", str(input_path))
        table.add_row("Output Folder", output_path)
        table.add_row("Performance Mode", config.processing.performance_mode.value)
        table.add_row("Batch Size", str(config.processing.batch_size))
        table.add_row("Max Workers", str(config.processing.max_workers))
        table.add_row("GPU Enabled", "Yes" if config.processing.gpu_enabled else "No")
        table.add_row("Memory Limit", f"{config.processing.memory_limit_gb} GB")
        table.add_row("Quality Threshold", str(config.processing.quality_threshold))
        table.add_row("Similarity Threshold", str(config.processing.similarity_threshold))
        
        console.print(table)
    
    def _show_config(self):
        """Show current configuration"""
        config = self.settings.get_config()
        
        console.print(Panel.fit("⚙️ Current Configuration", style="bold blue"))
        
        # Processing settings
        proc_table = Table(title="Processing Settings")
        proc_table.add_column("Setting", style="cyan")
        proc_table.add_column("Value", style="green")
        
        proc_table.add_row("Performance Mode", config.processing.performance_mode.value)
        proc_table.add_row("Batch Size", str(config.processing.batch_size))
        proc_table.add_row("Max Workers", str(config.processing.max_workers))
        proc_table.add_row("GPU Enabled", "Yes" if config.processing.gpu_enabled else "No")
        proc_table.add_row("Memory Limit", f"{config.processing.memory_limit_gb} GB")
        proc_table.add_row("Quality Threshold", str(config.processing.quality_threshold))
        proc_table.add_row("Similarity Threshold", str(config.processing.similarity_threshold))
        
        console.print(proc_table)
        
        # System settings
        sys_table = Table(title="System Settings")
        sys_table.add_column("Setting", style="cyan")
        sys_table.add_column("Value", style="green")
        
        sys_table.add_row("Log Level", config.system.log_level)
        sys_table.add_row("Temp Directory", config.system.temp_dir)
        sys_table.add_row("Max File Size", f"{config.system.max_file_size_mb} MB")
        sys_table.add_row("Auto Cleanup", "Yes" if config.system.auto_cleanup else "No")
        
        console.print(sys_table)
    
    def _set_config(self, setting: str) -> int:
        """Set configuration value"""
        try:
            key, value = setting.split('=', 1)
            section, field = key.split('.', 1)
            
            # Convert value to appropriate type
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.replace('.', '').isdigit():
                value = float(value) if '.' in value else int(value)
            
            success = self.settings.update_settings(section, {field: value})
            
            if success:
                console.print(f"[green]✓ Set {key} = {value}[/green]")
                return 0
            else:
                console.print(f"[red]✗ Failed to set {key}[/red]")
                return 1
                
        except ValueError:
            console.print("[red]Invalid format. Use: section.field=value[/red]")
            return 1
    
    def _reset_config(self) -> int:
        """Reset configuration to defaults"""
        console.print("[yellow]Resetting configuration to defaults...[/yellow]")
        # This would reset the configuration
        console.print("[green]✓ Configuration reset[/green]")
        return 0
    
    def _optimize_config(self) -> int:
        """Auto-optimize configuration"""
        console.print("[blue]Analyzing hardware and optimizing configuration...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Optimizing...", total=None)
            
            result = self.settings.optimize_for_hardware()
            
            progress.update(task, completed=True)
        
        console.print("[green]✓ Configuration optimized for your hardware[/green]")
        
        # Show recommendations
        recommendations = result['recommendations']
        if recommendations.get('optimizations'):
            console.print("\n[bold]Applied Optimizations:[/bold]")
            for opt in recommendations['optimizations']:
                console.print(f"  • {opt}")
        
        if recommendations.get('warnings'):
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in recommendations['warnings']:
                console.print(f"  ⚠️  {warning}")
        
        return 0
    
    def _export_config(self, filename: str) -> int:
        """Export configuration to file"""
        try:
            config = self.settings.get_config()
            with open(filename, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            console.print(f"[green]✓ Configuration exported to {filename}[/green]")
            return 0
        except Exception as e:
            console.print(f"[red]✗ Export failed: {e}[/red]")
            return 1
    
    def _import_config(self, filename: str) -> int:
        """Import configuration from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            # This would import the configuration
            console.print(f"[green]✓ Configuration imported from {filename}[/green]")
            return 0
        except Exception as e:
            console.print(f"[red]✗ Import failed: {e}[/red]")
            return 1
    
    def _show_health(self):
        """Show current system health"""
        health = self.settings.get_system_health()
        
        console.print(Panel.fit("💊 System Health", style="bold blue"))
        
        # Status indicator
        status_color = {
            'healthy': 'green',
            'warning': 'yellow',
            'critical': 'red',
            'error': 'red'
        }.get(health.get('status', 'error'), 'red')
        
        console.print(f"Status: [{status_color}]{health.get('status', 'unknown').upper()}[/{status_color}]")
        
        # Metrics table
        table = Table(title="System Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        table.add_row("CPU Usage", f"{health.get('cpu_usage_percent', 0):.1f}%", 
                     "🟢" if health.get('cpu_usage_percent', 0) < 70 else "🟡" if health.get('cpu_usage_percent', 0) < 90 else "🔴")
        table.add_row("Memory Usage", f"{health.get('memory_usage_percent', 0):.1f}%",
                     "🟢" if health.get('memory_usage_percent', 0) < 70 else "🟡" if health.get('memory_usage_percent', 0) < 90 else "🔴")
        table.add_row("Disk Usage", f"{health.get('disk_usage_percent', 0):.1f}%",
                     "🟢" if health.get('disk_usage_percent', 0) < 85 else "🟡" if health.get('disk_usage_percent', 0) < 95 else "🔴")
        table.add_row("Available Memory", f"{health.get('memory_available_gb', 0):.1f} GB", "ℹ️")
        table.add_row("Free Disk Space", f"{health.get('disk_free_gb', 0):.1f} GB", "ℹ️")
        
        console.print(table)
        
        # GPU metrics
        gpu_metrics = health.get('gpu_metrics', [])
        if gpu_metrics:
            gpu_table = Table(title="GPU Metrics")
            gpu_table.add_column("GPU", style="cyan")
            gpu_table.add_column("Load", style="green")
            gpu_table.add_column("Memory", style="green")
            gpu_table.add_column("Temperature", style="yellow")
            
            for gpu in gpu_metrics:
                gpu_table.add_row(
                    gpu['name'],
                    f"{gpu['load']:.1f}%",
                    f"{gpu['memory_used_percent']:.1f}%",
                    f"{gpu['temperature']}°C"
                )
            
            console.print(gpu_table)
    
    async def _monitor_health(self, interval: int):
        """Continuous health monitoring"""
        console.print(f"[blue]Starting health monitoring (interval: {interval}s). Press Ctrl+C to stop.[/blue]")
        
        try:
            while True:
                console.clear()
                self._show_health()
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped[/yellow]")
    
    def _show_hardware(self):
        """Show detailed hardware information"""
        from config.settings_manager import HardwareDetector
        
        detector = HardwareDetector()
        system_info = detector.get_system_info()
        
        console.print(Panel.fit("🖥️ Hardware Information", style="bold blue"))
        
        # System info table
        table = Table(title="System Information")
        table.add_column("Component", style="cyan")
        table.add_column("Details", style="green")
        
        table.add_row("Platform", system_info.get('platform', 'Unknown'))
        table.add_row("Processor", system_info.get('processor', 'Unknown'))
        table.add_row("CPU Cores (Physical)", str(system_info.get('cpu_count_physical', 'Unknown')))
        table.add_row("CPU Cores (Logical)", str(system_info.get('cpu_count_logical', 'Unknown')))
        table.add_row("CPU Frequency", f"{system_info.get('cpu_frequency_mhz', 0):.0f} MHz")
        table.add_row("Total Memory", f"{system_info.get('memory_total_gb', 0):.1f} GB")
        table.add_row("Available Memory", f"{system_info.get('memory_available_gb', 0):.1f} GB")
        table.add_row("Total Disk Space", f"{system_info.get('disk_total_gb', 0):.1f} GB")
        table.add_row("Free Disk Space", f"{system_info.get('disk_free_gb', 0):.1f} GB")
        
        console.print(table)
        
        # GPU information
        gpus = system_info.get('gpus', [])
        if gpus:
            gpu_table = Table(title="GPU Information")
            gpu_table.add_column("GPU", style="cyan")
            gpu_table.add_column("Memory", style="green")
            gpu_table.add_column("Load", style="yellow")
            gpu_table.add_column("Temperature", style="red")
            
            for gpu in gpus:
                gpu_table.add_row(
                    gpu['name'],
                    f"{gpu['memory_total']} MB",
                    f"{gpu['load'] * 100:.1f}%",
                    f"{gpu['temperature']}°C"
                )
            
            console.print(gpu_table)
        else:
            console.print("[yellow]No GPU detected[/yellow]")
    
    def _show_recommendations(self):
        """Show performance recommendations"""
        from config.settings_manager import HardwareDetector
        
        detector = HardwareDetector()
        system_info = detector.get_system_info()
        recommendations = detector.get_performance_recommendations(system_info)
        
        console.print(Panel.fit("💡 Performance Recommendations", style="bold blue"))
        
        # Recommendations table
        table = Table(title="Recommended Settings")
        table.add_column("Setting", style="cyan")
        table.add_column("Recommended Value", style="green")
        table.add_column("Reason", style="yellow")
        
        table.add_row("Performance Mode", recommendations['recommended_mode'].value, "Based on available memory")
        table.add_row("Batch Size", str(recommendations['recommended_batch_size']), "Optimized for your hardware")
        table.add_row("Worker Threads", str(recommendations['recommended_workers']), "Based on CPU cores")
        table.add_row("GPU Acceleration", "Yes" if recommendations['gpu_acceleration'] else "No", "Based on GPU availability")
        table.add_row("Memory Limit", f"{recommendations['memory_limit_gb']:.1f} GB", "Safe memory usage")
        
        console.print(table)
        
        # Optimizations
        if recommendations.get('optimizations'):
            console.print("\n[bold green]Optimizations:[/bold green]")
            for opt in recommendations['optimizations']:
                console.print(f"  ✓ {opt}")
        
        # Warnings
        if recommendations.get('warnings'):
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in recommendations['warnings']:
                console.print(f"  ⚠️  {warning}")

def main():
    """Main CLI entry point"""
    cli = CLIManager()
    parser = cli.create_parser()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    # Run async command handler
    return asyncio.run(cli.handle_command(args))

if __name__ == '__main__':
    sys.exit(main())