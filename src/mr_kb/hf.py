#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face Hub Integration for MindRoot Knowledge Bases

This module provides functions to upload and download KB exports to/from Hugging Face Hub.
"""

import os
import logging
import json
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

# Import huggingface_hub for Hub interactions
try:
    from huggingface_hub import HfApi, HfFolder, Repository, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from .kb_transfer import export_kb, import_kb

logger = logging.getLogger(__name__)


def check_hf_available():
    """Check if huggingface_hub is available and raise an error if not."""
    if not HF_AVAILABLE:
        raise ImportError(
            "The huggingface_hub package is required for Hugging Face integration. "
            "Please install it with 'pip install huggingface_hub'."
        )

async def upload_kb_to_hf(
    kb_instance, 
    repo_id: str, 
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload knowledge base export",
    progress_callback=None,
    temp_dir: Optional[str] = None,
    create_readme: bool = True
) -> str:
    """
    Export a knowledge base and upload it to Hugging Face Hub.
    
    Args:
        kb_instance: HierarchicalKnowledgeBase instance
        repo_id: Hugging Face repository ID (format: username/repo_name or org/repo_name)
        token: Hugging Face API token. If None, will look for token in cache or env var
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        progress_callback: Optional callback for progress updates
        temp_dir: Optional temporary directory for export files
        create_readme: Whether to create a README.md file with KB information
        
    Returns:
        URL of the uploaded repository
    """
    check_hf_available()
    
    # Initialize Hugging Face API
    api = HfApi(token=token)
    
    # Create a temporary directory for the export if not provided
    if temp_dir is None:
        temp_dir = f"kb_export_{int(time.time())}"
        os.makedirs(temp_dir, exist_ok=True)
    
    export_path = os.path.join(temp_dir, "kb_export.zip")
    
    try:
        # 1. Export the KB
        if progress_callback:
            progress_callback(0.1, "Exporting knowledge base")
        
        await export_kb(kb_instance, export_path, 
                       progress_callback=lambda p, m=None: progress_callback(p * 0.6, m) if progress_callback else None)
        
        # 2. Create or get the repository
        if progress_callback:
            progress_callback(0.7, "Preparing Hugging Face repository")
        
        try:
            # Check if repo exists
            api.repo_info(repo_id=repo_id)
        except RepositoryNotFoundError:
            # Create new repo if it doesn't exist
            create_repo(repo_id, private=private, token=token, repo_type="dataset")
        
        # 3. Create README.md with KB information if requested
        if create_readme:
            if progress_callback:
                progress_callback(0.75, "Creating README")
            
            # Extract metadata from the export
            import zipfile
            with zipfile.ZipFile(export_path, 'r') as zipf:
                try:
                    with zipf.open("kb_metadata.json") as f:
                        kb_metadata = json.load(f)
                except (KeyError, json.JSONDecodeError):
                    kb_metadata = {}
            
            # Create README content
            readme_content = f"# MindRoot Knowledge Base Export\n\n"
            readme_content += f"This repository contains a MindRoot knowledge base export.\n\n"
            
            # Add metadata if available
            if kb_metadata:
                readme_content += f"## Metadata\n\n"
                readme_content += f"- Export Version: {kb_metadata.get('export_version', 'Unknown')}\n"
                readme_content += f"- Export Date: {kb_metadata.get('export_date', 'Unknown')}\n"
                
                # Add component information
                if 'components' in kb_metadata:
                    readme_content += f"\n### Components\n\n"
                    for component, exists in kb_metadata['components'].items():
                        readme_content += f"- {component}: {'✅' if exists else '❌'}\n"
            
            # Add usage instructions
            readme_content += f"\n## Usage\n\n"
            readme_content += f"This knowledge base can be imported into a MindRoot instance using:\n\n"
            readme_content += f"```python\n"
            readme_content += f"from mr_kb import import_kb_from_hf\n"
            readme_content += f"\n"
            readme_content += f"# Initialize your KB instance\n"
            readme_content += f"kb = HierarchicalKnowledgeBase(persist_dir='/path/to/kb')\n"
            readme_content += f"\n"
            readme_content += f"# Import from Hugging Face\n"
            readme_content += f"await import_kb_from_hf(kb, repo_id='{repo_id}')\n"
            readme_content += f"```\n"
            
            # Write README to file
            readme_path = os.path.join(temp_dir, "README.md")
            with open(readme_path, 'w') as f:
                f.write(readme_content)
        
        # 4. Upload files to the repository
        if progress_callback:
            progress_callback(0.8, "Uploading to Hugging Face Hub")
        
        # Upload the export file
        api.upload_file(
            path_or_fileobj=export_path,
            path_in_repo="kb_export.zip",
            repo_id=repo_id,
            token=token,
            commit_message=commit_message
        )
        
        # Upload README if created
        if create_readme:
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                token=token,
                commit_message="Update README"
            )
        
        # 5. Create dataset card metadata
        if progress_callback:
            progress_callback(0.9, "Creating dataset card")
        
        # Create dataset-metadata.json for better HF Hub integration
        metadata = {
            "library_name": "mindroot",
            "tags": ["mindroot", "knowledge-base", "embeddings", "vector-database"],
            "language": "en",
            "license": "mit",  # Default license, can be customized
        }
        
        metadata_path = os.path.join(temp_dir, "dataset-metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Upload metadata
        api.upload_file(
            path_or_fileobj=metadata_path,
            path_in_repo="dataset-metadata.json",
            repo_id=repo_id,
            token=token,
            commit_message="Add dataset metadata"
        )
        
        if progress_callback:
            progress_callback(1.0, "Upload complete")
        
        # Return the repository URL
        return f"https://huggingface.co/datasets/{repo_id}"
        
    except Exception as e:
        logger.error(f"Failed to upload KB to Hugging Face: {str(e)}")
        raise

async def download_kb_from_hf(
    kb_instance, 
    repo_id: str, 
    token: Optional[str] = None,
    revision: Optional[str] = None,
    progress_callback=None,
    temp_dir: Optional[str] = None
) -> bool:
    """
    Download a knowledge base from Hugging Face Hub and import it.
    
    Args:
        kb_instance: HierarchicalKnowledgeBase instance
        repo_id: Hugging Face repository ID (format: username/repo_name or org/repo_name)
        token: Hugging Face API token. If None, will look for token in cache or env var
        revision: Specific revision to download (branch, tag, or commit hash)
        progress_callback: Optional callback for progress updates
        temp_dir: Optional temporary directory for downloaded files
        
    Returns:
        True if successful
    """
    check_hf_available()
    
    # Initialize Hugging Face API
    api = HfApi(token=token)
    
    # Create a temporary directory for the download if not provided
    if temp_dir is None:
        temp_dir = f"kb_download_{int(time.time())}"
        os.makedirs(temp_dir, exist_ok=True)
    
    download_path = os.path.join(temp_dir, "kb_export.zip")
    
    try:
        # 1. Check if the repository exists
        if progress_callback:
            progress_callback(0.1, "Checking repository")
        
        try:
            repo_info = api.repo_info(repo_id=repo_id, revision=revision)
        except RepositoryNotFoundError:
            raise ValueError(f"Repository {repo_id} not found on Hugging Face Hub")
        except RevisionNotFoundError:
            raise ValueError(f"Revision {revision} not found in repository {repo_id}")
        
        # 2. Download the KB export file
        if progress_callback:
            progress_callback(0.2, "Downloading knowledge base export")
        
        try:
            # Use the HF API to download the file
            api.hf_hub_download(
                repo_id=repo_id,
                filename="kb_export.zip",
                revision=revision,
                local_dir=temp_dir,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            raise ValueError(f"Failed to download KB export: {str(e)}")
        
        # 3. Import the KB
        if progress_callback:
            progress_callback(0.4, "Importing knowledge base")
        
        await import_kb(kb_instance, download_path, 
                      progress_callback=lambda p, m=None: progress_callback(0.4 + p * 0.6, m) if progress_callback else None)
        
        if progress_callback:
            progress_callback(1.0, "Import complete")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download KB from Hugging Face: {str(e)}")
        raise
    finally:
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir) and temp_dir.startswith("kb_download_"):
            shutil.rmtree(temp_dir)

async def list_kb_repos(
    username_or_org: Optional[str] = None,
    token: Optional[str] = None,
    filter_tags: List[str] = ["mindroot", "knowledge-base"]
) -> List[Dict]:
    """
    List knowledge base repositories on Hugging Face Hub.
    
    Args:
        username_or_org: Optional username or organization to filter by
        token: Hugging Face API token. If None, will look for token in cache or env var
        filter_tags: Tags to filter repositories by
        
    Returns:
        List of repository information dictionaries
    """
    check_hf_available()
    
    # Initialize Hugging Face API
    api = HfApi(token=token)
    
    try:
        # Get datasets with the specified tags
        datasets = api.list_datasets(
            author=username_or_org,
            filter=filter_tags
        )
        
        # Format the results
        results = []
        for dataset in datasets:
            results.append({
                "repo_id": dataset.id,
                "name": dataset.id.split("/")[-1] if "/" in dataset.id else dataset.id,
                "author": dataset.author,
                "tags": dataset.tags,
                "last_modified": dataset.lastModified,
                "private": dataset.private,
                "url": f"https://huggingface.co/datasets/{dataset.id}"
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to list KB repositories: {str(e)}")
        raise


def create_command_handlers():
    """
    Create command handlers for the Hugging Face integration.
    
    This function creates command handlers that can be registered with the MindRoot
    command system to provide HF Hub integration for KB export/import.
    
    Returns:
        Dictionary of command handlers
    """
    from lib.providers.commands import command
    
    @command()
    async def upload_kb_to_huggingface(repo_id, token=None, private=False, context=None):
        """
        Export a knowledge base and upload it to Hugging Face Hub.
        
        Args:
            repo_id: Hugging Face repository ID (format: username/repo_name or org/repo_name)
            token: Hugging Face API token. If None, will look for token in cache or env var
            private: Whether to create a private repository
            
        Returns:
            URL of the uploaded repository
        """
        if not context or not hasattr(context, 'kb'):
            return {"error": "No knowledge base available in context"}
            
        kb_instance = context.kb
        
        try:
            result = await upload_kb_to_hf(kb_instance, repo_id, token, private)
            return {"success": True, "url": result}
        except Exception as e:
            return {"error": str(e)}
    
    @command()
    async def download_kb_from_huggingface(repo_id, token=None, revision=None, context=None):
        """
        Download a knowledge base from Hugging Face Hub and import it.
        
        Args:
            repo_id: Hugging Face repository ID (format: username/repo_name or org/repo_name)
            token: Hugging Face API token. If None, will look for token in cache or env var
            revision: Specific revision to download (branch, tag, or commit hash)
            
        Returns:
            Success status
        """
        if not context or not hasattr(context, 'kb'):
            return {"error": "No knowledge base available in context"}
            
        kb_instance = context.kb
        
        try:
            result = await download_kb_from_hf(kb_instance, repo_id, token, revision)
            return {"success": result}
        except Exception as e:
            return {"error": str(e)}
    
    @command()
    async def list_kb_repos_on_huggingface(username_or_org=None, token=None):
        """
        List knowledge base repositories on Hugging Face Hub.
        
        Args:
            username_or_org: Optional username or organization to filter by
            token: Hugging Face API token. If None, will look for token in cache or env var
            
        Returns:
            List of repository information dictionaries
        """
        try:
            result = await list_kb_repos(username_or_org, token)
            return {"success": True, "repositories": result}
        except Exception as e:
            return {"error": str(e)}
    
    return {
        "upload_kb_to_huggingface": upload_kb_to_huggingface,
        "download_kb_from_huggingface": download_kb_from_huggingface,
        "list_kb_repos_on_huggingface": list_kb_repos_on_huggingface
    }

# Command-line interface for standalone usage
def main():
    """
    Command-line interface for Hugging Face Hub integration.
    
    Usage:
        python -m mr_kb.hf upload --kb-dir=/path/to/kb --repo-id=username/repo-name [--token=YOUR_TOKEN] [--private]
        python -m mr_kb.hf download --kb-dir=/path/to/kb --repo-id=username/repo-name [--token=YOUR_TOKEN] [--revision=main]
        python -m mr_kb.hf list [--username=username] [--token=YOUR_TOKEN]
    """
    import argparse
    import asyncio
    from .kb import HierarchicalKnowledgeBase
    
    parser = argparse.ArgumentParser(description="MindRoot Knowledge Base Hugging Face Integration")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload a knowledge base to Hugging Face Hub")
    upload_parser.add_argument("--kb-dir", required=True, help="Path to the knowledge base directory")
    upload_parser.add_argument("--repo-id", required=True, help="Hugging Face repository ID (username/repo-name)")
    upload_parser.add_argument("--token", help="Hugging Face API token")
    upload_parser.add_argument("--private", action="store_true", help="Create a private repository")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a knowledge base from Hugging Face Hub")
    download_parser.add_argument("--kb-dir", required=True, help="Path to the knowledge base directory")
    download_parser.add_argument("--repo-id", required=True, help="Hugging Face repository ID (username/repo-name)")
    download_parser.add_argument("--token", help="Hugging Face API token")
    download_parser.add_argument("--revision", help="Specific revision to download (branch, tag, or commit hash)")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List knowledge base repositories on Hugging Face Hub")
    list_parser.add_argument("--username", help="Username or organization to filter by")
    list_parser.add_argument("--token", help="Hugging Face API token")
    
    args = parser.parse_args()
    
    # Define progress callback
    def progress_callback(progress, message=None):
        if message:
            print(f"[{progress:.1%}] {message}")
        else:
            print(f"[{progress:.1%}]")
    
    if args.command == "upload":
        # Initialize KB
        kb = HierarchicalKnowledgeBase(persist_dir=args.kb_dir)
        
        # Run upload
        try:
            result = asyncio.run(upload_kb_to_hf(
                kb, args.repo_id, args.token, args.private, 
                progress_callback=progress_callback
            ))
            print(f"Upload completed successfully: {result}")
            return 0
        except Exception as e:
            print(f"Upload failed: {str(e)}")
            return 1
    
    elif args.command == "download":
        # Initialize KB
        kb = HierarchicalKnowledgeBase(persist_dir=args.kb_dir)
        
        # Run download
        try:
            result = asyncio.run(download_kb_from_hf(
                kb, args.repo_id, args.token, args.revision,
                progress_callback=progress_callback
            ))
            print(f"Download completed successfully")
            return 0
        except Exception as e:
            print(f"Download failed: {str(e)}")
            return 1
    
    elif args.command == "list":
        # Run list
        try:
            result = asyncio.run(list_kb_repos(args.username, args.token))
            print(f"Found {len(result)} knowledge base repositories:")
            for repo in result:
                print(f"- {repo['repo_id']} by {repo['author']} ({'private' if repo['private'] else 'public'})")
                print(f"  URL: {repo['url']}")
                print(f"  Last modified: {repo['last_modified']}")
                print(f"  Tags: {', '.join(repo['tags'])}")
                print()
            return 0
        except Exception as e:
            print(f"List failed: {str(e)}")
            return 1
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
