"""Script to run the RAPTOR monitoring dashboard."""

import logging
import time
from pathlib import Path
import argparse
from src.monitoring.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)

def setup_logging(log_file: Path = None):
    """Configure logging."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format
    )
    
    # Add file handler if log file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

def main():
    """Run the monitoring dashboard."""
    parser = argparse.ArgumentParser(description='RAPTOR System Monitor')
    parser.add_argument(
        '--base-dir',
        type=str,
        default=str(Path.cwd()),
        help='Base directory for the RAPTOR system'
    )
    parser.add_argument(
        '--refresh-interval',
        type=int,
        default=60,
        help='Dashboard refresh interval in seconds'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Custom path to save the dashboard HTML'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file'
    )
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    
    # Setup logging
    log_file = Path(args.log_file) if args.log_file else base_dir / "data" / "logs" / "monitor.log"
    setup_logging(log_file)
    
    logger.info("Starting RAPTOR System Monitor...")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Refresh interval: {args.refresh_interval} seconds")
    logger.info(f"Log file: {log_file}")
    
    # Initialize monitor
    monitor = SystemMonitor(base_dir)
    
    try:
        while True:
            logger.info("=== Updating Monitoring Dashboard ===")
            
            # Update system metrics
            logger.info("Updating system metrics...")
            monitor.update_system_metrics()
            
            # Generate dashboard
            logger.info("Generating dashboard...")
            output_path = Path(args.output_path) if args.output_path else None
            monitor.generate_dashboard(output_path)
            
            # Get and log system health
            health = monitor.get_system_health()
            logger.info("=== System Health Status ===")
            logger.info(f"Status: {health['status']}")
            logger.info(f"Processing Rate: {health['processing_rate']:.2f}%")
            logger.info(f"Cache Efficiency: {health['cache_efficiency']:.2f}%")
            logger.info("System Load:")
            for resource, usage in health['system_load'].items():
                logger.info(f"  {resource}: {usage:.2f}%")
            
            # Wait for next refresh
            logger.info(f"Waiting {args.refresh_interval} seconds until next update...")
            time.sleep(args.refresh_interval)
            
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitoring failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
