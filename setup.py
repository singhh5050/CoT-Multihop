"""
Setup script for multi-hop evaluation environment

This script sets up the environment for evaluating multi-hop question answering
models using the MuSiQue dataset and OpenAI API.
"""

import os
import sys
from datasets import load_dataset
from dotenv import load_dotenv

def setup_dataset(split="validation"):
    """
    Load the MuSiQue dataset.
    
    Args:
        split: Dataset split to load (default: validation)
    
    Returns:
        The loaded dataset
    """
    try:
        # Load MuSiQue dataset
        dataset = load_dataset('fladhak/musique', split=split, trust_remote_code=True)
        print(f"Successfully loaded MuSiQue dataset ({split} split) with {len(dataset)} examples")
        return dataset
    except Exception as e:
        print(f"Error loading MuSiQue dataset: {str(e)}")
        return None

def setup_api_key():
    """
    Configure the OpenAI API key from environment variables.
    
    Returns:
        Boolean indicating success
    """
    # First try loading from .env file
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set")
        print("Please set it in your environment or create a .env file")
        return False
    
    # Verify the API key format (basic check)
    if not api_key.startswith('sk-'):
        print("WARNING: The API key format doesn't look valid")
        print("API keys should start with 'sk-'")
    
    print("OpenAI API key configured successfully")
    return True

def setup_directories():
    """
    Create necessary directories for results.
    
    Returns:
        Boolean indicating success
    """
    try:
        # Create directories for results
        for directory in ['results/musique', 'results/metrics', 'results/by_hop', 'results/visualizations']:
            os.makedirs(directory, exist_ok=True)
        
        print("Directory structure created successfully")
        return True
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
        return False

def main():
    """Main function to run all setup steps."""
    print("Setting up multi-hop QA evaluation environment...")
    
    # Setup API key
    if not setup_api_key():
        return 1
    
    # Setup directories
    if not setup_directories():
        return 1
    
    # Test dataset access
    dataset = setup_dataset()
    if dataset is None:
        return 1
    
    # Display an example
    try:
        example = dataset[0]
        print("\nExample from MuSiQue dataset:")
        print(f"Question: {example['question']}")
        print(f"Answer: {example['answer']}")
        print(f"Hop count: {len(example['question_decomposition'])}")
    except Exception as e:
        print(f"Error accessing dataset example: {str(e)}")
    
    print("\nSetup completed successfully.")
    return 0

if __name__ == '__main__':
    sys.exit(main())