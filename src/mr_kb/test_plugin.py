import asyncio
import os
from dotenv import load_dotenv
from .vector_only_kb import HierarchicalKnowledgeBase
from .utils import format_supported_types

async def test_kb():
    """Test the knowledge base plugin functionality."""
    try:
        # Load API key from environment
        load_dotenv()
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("Please set OPENAI_API_KEY in your environment or .env file")

        # Initialize KB
        kb = HierarchicalKnowledgeBase(persist_dir="./test_storage")
        
        # Show supported file types
        print("\nFile type support:")
        print(format_supported_types(kb.supported_types))
        
        # Create test data directory
        os.makedirs("./test_data", exist_ok=True)
        
        # Create a test file
        with open("./test_data/test.txt", "w") as f:
            f.write("This is a test document for the knowledge base.")

        # Create index
        print("\nCreating index...")
        await kb.create_index("./test_data")
        
        # Test query
        print("\nTesting query...")
        context = await kb.get_relevant_context(
            "What is this document about?",
            format_type="detailed"
        )
        print(context)

    except Exception as e:
        print(f"Error in test: {str(e)}")
        raise
    finally:
        # Clean up test files
        if os.path.exists("./test_data"):
            import shutil
            shutil.rmtree("./test_data")
        if os.path.exists("./test_storage"):
            shutil.rmtree("./test_storage")

if __name__ == "__main__":
    asyncio.run(test_kb())
