import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheAnalyzer:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise ValueError(f"Cache directory {cache_dir} does not exist")
        
    def analyze_database(self, db_path: Path) -> Dict:
        """Analyze a single SQLite database file."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            stats = {
                "file_size_mb": db_path.stat().st_size / (1024 * 1024),
                "last_modified": datetime.fromtimestamp(db_path.stat().st_mtime).isoformat(),
                "tables": {}
            }
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Get sample of columns
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                
                stats["tables"][table_name] = {
                    "row_count": row_count,
                    "columns": columns
                }
            
            conn.close()
            return stats
            
        except sqlite3.Error as e:
            logger.error(f"Error analyzing database {db_path}: {str(e)}")
            return {"error": str(e)}

    def analyze_tree_statistics(self, stats_file: Path) -> Dict:
        """Analyze a tree statistics JSON file."""
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            # Extract key metrics
            timestamp = int(stats_file.stem.split('_')[1])
            return {
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                "file_size": stats_file.stat().st_size,
                "content": stats
            }
        except Exception as e:
            logger.error(f"Error analyzing tree statistics {stats_file}: {str(e)}")
            return {"error": str(e)}

    def get_latest_tree_stats(self) -> Optional[Dict]:
        """Get the most recent tree statistics file and its contents."""
        tree_files = list(self.cache_dir.glob("tree_*_statistics.json"))
        if not tree_files:
            return None
        
        latest_file = max(tree_files, key=lambda x: int(x.stem.split('_')[1]))
        return self.analyze_tree_statistics(latest_file)

    def analyze_all(self) -> Dict:
        """Analyze all cache components and generate a comprehensive report."""
        report = {
            "databases": {},
            "tree_statistics": [],
            "summary": {}
        }
        
        # Analyze all databases
        for db_file in self.cache_dir.glob("*.db"):
            report["databases"][db_file.name] = self.analyze_database(db_file)
        
        # Analyze all tree statistics
        tree_files = list(self.cache_dir.glob("tree_*_statistics.json"))
        for stats_file in sorted(tree_files, key=lambda x: int(x.stem.split('_')[1])):
            report["tree_statistics"].append(self.analyze_tree_statistics(stats_file))
        
        # Generate summary
        latest_tree_stats = self.get_latest_tree_stats()
        report["summary"] = {
            "total_databases": len(report["databases"]),
            "total_tree_stats_files": len(report["tree_statistics"]),
            "latest_tree_update": latest_tree_stats["datetime"] if latest_tree_stats else None,
            "total_cache_size_mb": sum(
                db["file_size_mb"] 
                for db in report["databases"].values() 
                if "file_size_mb" in db
            )
        }
        
        return report

def print_report(report: Dict):
    """Print a human-readable version of the analysis report."""
    print("\n=== Cache Analysis Report ===\n")
    
    print("Summary:")
    print(f"- Total databases: {report['summary']['total_databases']}")
    print(f"- Total tree statistics files: {report['summary']['total_tree_stats_files']}")
    print(f"- Latest tree update: {report['summary']['latest_tree_update']}")
    print(f"- Total cache size: {report['summary']['total_cache_size_mb']:.2f} MB")
    
    print("\nDatabases:")
    for db_name, db_stats in report["databases"].items():
        print(f"\n{db_name}:")
        if "error" in db_stats:
            print(f"  Error: {db_stats['error']}")
            continue
            
        print(f"  Size: {db_stats['file_size_mb']:.2f} MB")
        print(f"  Last modified: {db_stats['last_modified']}")
        print("  Tables:")
        for table_name, table_stats in db_stats["tables"].items():
            print(f"    - {table_name}: {table_stats['row_count']} rows")
    
    if report["tree_statistics"]:
        print("\nTree Statistics Files:")
        for stats in report["tree_statistics"][-5:]:  # Show last 5 only
            print(f"\n{stats['datetime']}:")
            if "error" in stats:
                print(f"  Error: {stats['error']}")
                continue
            print(f"  File size: {stats['file_size']} bytes")

if __name__ == "__main__":
    cache_dir = Path(__file__).parents[2] / "cache"
    analyzer = CacheAnalyzer(str(cache_dir))
    report = analyzer.analyze_all()
    print_report(report)
