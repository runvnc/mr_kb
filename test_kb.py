import asyncio
import os
from dotenv import load_dotenv
from vector_only_kb import HierarchicalKnowledgeBase as VectorKnowledgeBase
from utils import format_supported_types
import shutil

async def main():
    """Test the HierarchicalKnowledgeBase with different file types."""
    try:
        # Load API key from environment
        load_dotenv()
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("Please set OPENAI_API_KEY in your environment or .env file")

        try:
            # Create test data directory
            os.makedirs("./data", exist_ok=True)
            
            # Create test files of different types
            with open("./data/test.txt", "w") as f:
                f.write("This is a text file for testing.")
                
            # Create a simple Excel file if pandas is available
            try:
                import pandas as pd
                df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
                df.to_excel("./data/test.xlsx", index=False)
            except ImportError:
                print("Excel support not available (pandas not installed)")
                
            # Create a Word document if python-docx is available
            try:
                from docx import Document
                doc = Document()
                doc.add_paragraph("This is a test Word document.")
                doc.save("./data/test.docx")
            except ImportError:
                print("Word document support not available (python-docx not installed)")

            # Initialize KB
            # kb = HierarchicalKnowledgeBase(persist_dir="./storage")  # Hybrid search
            kb = VectorKnowledgeBase(persist_dir="./storage")  # Vector-only search
            
            # Show supported file types
            print("\nFile type support:")
            print(format_supported_types(kb.supported_types))
            
            # Create index
            print("\nCreating index...")
            await kb.create_index("./data")
            
            # Show indexed documents
            print("\nIndexed documents:")
            docs = kb.get_document_info()
            for doc in docs:
                print(f"\nDocument: {doc['file_path']}")
                #print(f"Total nodes: {doc['total_nodes']}")
                #print("Nodes per level:")
                #for size, count in doc['level_counts'].items():
                #    print(f"  Chunk size {size}: {count} nodes")
            return 
            total_time = 0
            creation_queries = 0
            reuse_queries = 0
            # Test multiple retrievals
            test_queries = [
                "What types of files are in the test data?",
                "Tell me about the text file contents",
                "What's in the Excel file?",
                "Describe the Word document",
                "What types of files are supported?",
                "What's the property address?"
            ]
            
            print("\nTesting multiple retrievals...")
            for i, query in enumerate(test_queries, 1):
                print(f"\nQuery {i}: {query}")
                context, stats = await kb.get_relevant_context(
                    query,
                    format_type="detailed"
                )
                print(context)
                # Skip printing context, just show query number and stats
                print("=" * 40 + "\n")
                
                # Accumulate statistics
                total_time += stats['total_time'].total_seconds()
                if stats['retriever_creation']:
                    creation_queries += 1
                else:
                    reuse_queries += 1
                
            # Print timing statistics
            print("\nRetrieval Statistics:")
            print("=" * 40)
            print(f"Average time per query: {total_time/len(test_queries):.3f} seconds")
            print(f"Queries with retriever creation: {creation_queries}")
            print(f"Queries with retriever reuse: {reuse_queries}")
            
            
        finally:
            print("Done.")
            # Clean up test files
            #if os.path.exists("./data"):
            #    shutil.rmtree("./data")
            #if os.path.exists("./storage"):
        #    shutil.rmtree("./storage")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
