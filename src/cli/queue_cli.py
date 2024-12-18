#!/usr/bin/env python3
"""Command line interface for managing the document processing queue."""

import argparse
import sys
import json
from typing import List, Dict, Optional
from pathlib import Path

from src.processing.queue_manager import DocumentQueueManager, QueuePriority
from src.utils.config import load_config

class QueueCLI:
    """CLI for managing document processing queue."""
    
    def __init__(self):
        """Initialize CLI with default configuration."""
        self.config = load_config()
        self.queue_manager = DocumentQueueManager(self.config)
    
    def status(self) -> Dict:
        """Get current queue status."""
        return {
            'queue_size': self.queue_manager.queue_size(),
            'failed_queue_size': self.queue_manager.failed_queue_size(),
            'processing_rate': self.queue_manager.get_processing_rate(),
            'error_rate': self.queue_manager.get_error_rate()
        }
    
    def list_failed(self) -> List[Dict]:
        """List documents in failed queue."""
        return self.queue_manager.list_failed_documents()
    
    def retry_failed(self, doc_id: Optional[str] = None) -> int:
        """Retry failed documents."""
        return self.queue_manager.retry_failed_documents(doc_id)
    
    def prioritize(self, doc_id: str, priority: str) -> bool:
        """Change document priority."""
        try:
            priority_enum = QueuePriority[priority.upper()]
            return self.queue_manager.set_priority(doc_id, priority_enum)
        except KeyError:
            print(f"Invalid priority: {priority}. Use HIGH, NORMAL, or LOW")
            return False

def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description='Document Queue Management CLI')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Status command
    subparsers.add_parser('status', help='Show queue status')
    
    # List failed documents
    subparsers.add_parser('list-failed', help='List failed documents')
    
    # Retry failed documents
    retry_parser = subparsers.add_parser('retry', help='Retry failed documents')
    retry_parser.add_argument('--doc-id', help='Specific document ID to retry')
    
    # Change priority
    priority_parser = subparsers.add_parser('prioritize', help='Change document priority')
    priority_parser.add_argument('doc_id', help='Document ID')
    priority_parser.add_argument('priority', choices=['high', 'normal', 'low'], 
                               help='New priority level')
    
    args = parser.parse_args()
    cli = QueueCLI()
    
    if args.command == 'status':
        status = cli.status()
        print(json.dumps(status, indent=2))
    
    elif args.command == 'list-failed':
        failed = cli.list_failed()
        print(json.dumps(failed, indent=2))
    
    elif args.command == 'retry':
        retried = cli.retry_failed(args.doc_id)
        print(f"Retried {retried} documents")
    
    elif args.command == 'prioritize':
        success = cli.prioritize(args.doc_id, args.priority)
        if success:
            print(f"Updated priority for document {args.doc_id}")
        else:
            print(f"Failed to update priority for document {args.doc_id}")
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
