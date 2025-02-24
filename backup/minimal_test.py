import asyncio
import os
from dotenv import load_dotenv
from hierarchical_based import HierarchicalKnowledgeBase

async def test():
    try:
        # Load API key from environment
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")

        # Create a test directory and file
        os.makedirs("test_data", exist_ok=True)
        with open("test_data/test.txt", "w") as f:
            f.write("This is a test document. It contains multiple sentences. ")
            f.write("We will use this to verify the knowledge base functionality. ")
            f.write("Each sentence should be indexed separately in the hierarchy.")

        # Initialize KB
        kb = HierarchicalKnowledgeBase(
            persist_dir="test_storage",
            chunk_sizes=[512, 128, 32],  # Smaller chunks for our test
            openai_api_key=api_key
        )

        # Create index
        print("Creating index...")
        await kb.create_index("test_data")

        # Get document info
        print("\nDocument info:")
        docs = kb.get_document_info()
        for doc in docs:
            print(f"\nFilename: {doc['filename']}")
            print(f"Total nodes: {doc['total_nodes']}")
            print("Nodes per level:")
            for size, count in doc['level_counts'].items():
                print(f"  Chunk size {size}: {count} nodes")

        # Test retrieval
        print("\nTesting retrieval...")
        context = await kb.get_relevant_context(
            "What does this document contain?",
            format_type="detailed"
        )
        print("\nRetrieved context:")
        print(context)

        # Cleanup
        print("\nCleaning up...")
        import shutil
        shutil.rmtree("test_data")
        shutil.rmtree("test_storage")
        print("Test complete!")

    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test())
