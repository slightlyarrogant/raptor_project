"""
Test Runner for RAPTOR
Executes all tests with proper setup and teardown.
"""

import unittest
import logging
import sys
import os
from dotenv import load_dotenv
import time
from tests.setup_test_env import setup_test_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_tests():
    """Run all test suites."""
    try:
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded")
        
        # Setup test environment
        logger.info("Setting up test environment...")
        if not setup_test_index():
            logger.error("Failed to setup test environment")
            return False
        
        logger.info("Waiting for index to be ready...")
        time.sleep(40)  # Wait for Pinecone index to be fully ready
        
        # Discover and run tests
        logger.info("Running tests...")
        
        # Create test suite with proper path handling
        loader = unittest.TestLoader()
        start_dir = os.path.dirname(os.path.abspath(__file__))
        suite = loader.discover(
            start_dir, 
            pattern="test_*.py",
            top_level_dir=os.path.dirname(start_dir)  # Set top level to project root
        )
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Print summary
        logger.info("\nTest Summary:")
        logger.info("-" * 50)
        logger.info(f"Tests run: {result.testsRun}")
        logger.info(f"Failures: {len(result.failures)}")
        logger.info(f"Errors: {len(result.errors)}")
        logger.info(f"Skipped: {len(result.skipped)}")
        
        return len(result.failures) == 0 and len(result.errors) == 0
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 