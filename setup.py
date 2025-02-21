from setuptools import setup, find_packages

setup(
    name="kb",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core requirements
        "llama-index==0.9.48",
        "python-dotenv>=0.19.0",
        "openai>=1.0.0",
        "nltk>=3.6.0",
        "python-magic>=0.4.24",
        "tiktoken>=0.5.0",
        "rank_bm25>=0.2.2",
        
        # Document format support
        "PyPDF2>=3.0.0",
        "pandas>=2.0.0",
        "openpyxl>=3.0.0",
        "python-docx>=0.8.11",
    ],
    python_requires='>=3.8',
    author="Your Name",
    author_email="your.email@example.com",
    description="A hierarchical knowledge base with hybrid search and multi-format document support",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)