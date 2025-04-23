from setuptools import setup, find_packages

setup(
    name="mr_kb",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "llama-index>=0.12.20",
        "python-dotenv>=0.19.0",
        "openai>=1.0.0",
        "nltk>=3.6.0",
        "python-magic>=0.4.24",
        "tiktoken>=0.5.0",
        "rank_bm25>=0.2.2",
        "jsonpickle",
        "pandas>=2.0.0",
        "openpyxl>=3.0.0",
        "python-docx>=0.8.11",
        "PyPDF2>=3.0.0",
        "llama-index-vector-stores-chroma"
    ],
    package_data={
        "mr_kb": [
            "templates/*.jinja2",
            "inject/*.jinja2",
            "static/*.js",
            "static/js/*.js",
            "static/css/*.css",
            "static/*.md"
        ],
    },
    python_requires='>=3.8',
    author="MindRoot",
    author_email="info@mindroot.ai",
    description="A hierarchical knowledge base with vector search for MindRoot",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
