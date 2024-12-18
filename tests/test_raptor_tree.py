import logging

logger = logging.getLogger(__name__)

class TestRaptorTree:
    def setUp(self):
        """Set up test fixtures with real services."""
        logger.info("Initializing RaptorTree with real services...")
        # ... existing setup code ...

    def tearDown(self):
        """Clean up after each test."""
        logger.info("Cleaning up test resources...")
        try:
            # Clean up Pinecone namespaces
            if hasattr(self, 'index'):
                self.index.delete(
                    deleteAll=True,
                    namespace=f"{self.config['storage']['namespace']}_chunks"
                )
                self.index.delete(
                    deleteAll=True,
                    namespace=f"{self.config['storage']['namespace']}_summaries"
                )
                logger.info("✓ Cleaned up Pinecone vectors")
                
            # Reset any other stateful components
            if hasattr(self, 'raptor'):
                self.raptor = None
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}") 