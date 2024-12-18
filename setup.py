from setuptools import setup, find_packages

setup(
    name="raptor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "numpy>=1.24.2",
        "scikit-learn>=1.2.2",
        "umap-learn>=0.5.3",
        "nltk>=3.8.1",
        
        # Visualization
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "plotly>=5.13.1",
        "networkx>=3.0",
        
        # Other utilities
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0"
    ],
    extras_require={
        'test': [
            'pytest>=7.3.1',
            'pytest-cov>=4.0.0'
        ]
    }
) 