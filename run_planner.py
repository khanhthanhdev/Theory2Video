
import asyncio
import os
import shutil
import traceback
from generate_video import EnhancedVideoGenerator, VideoGenerationConfig
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file if present

async def main():
    """
    This script runs the full video generation pipeline with a hardcoded, 
    multi-provider LLM configuration.

    EDIT THE CONFIGURATION AND INPUTS SECTIONS BELOW TO TEST.
    """
    print("--- Starting Self-Contained Multi-Provider Test ---")

    # --- 1. Hardcoded Configuration ---
    # EDIT these values to change the models and providers.
    # --------------------------------------------------------------------------
    PLANNER_MODEL = "openai/o3-mini" # e.g., "openai/gpt-4o", "gemini/gemini-1.5-pro"
    HELPER_MODEL = "openai/o3-mini" # e.g., "openai/gpt-4o-mini", "bedrock/anthropic.claude-3-sonnet..."

    # --- Credentials ---
    # IMPORTANT: Set the credentials for the providers you are using.
    # You can paste them directly or use environment variables.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "PASTE_YOUR_GEMINI_KEY_HERE")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "PASTE_YOUR_OPENAI_KEY_HERE")
    
    # For AWS Bedrock
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "PASTE_YOUR_AWS_ACCESS_KEY_ID_HERE")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "PASTE_YOUR_AWS_SECRET_ACCESS_KEY_HERE")
    AWS_REGION_NAME = os.getenv("AWS_REGION_NAME", "ap-southeast-1")
    # --------------------------------------------------------------------------

    # --- 2. Inputs ---
    # EDIT these values to change the video content.
    # --------------------------------------------------------------------------
    topic = "Gradient Descent"
    description = "Gradient Descent is a fundamental optimization algorithm used in machine learning and deep learning to minimize a cost function (or loss function). It iteratively adjusts the model's parameters to reduce the error between predicted and actual values."
    output_dir = "live_test_output"
    # --------------------------------------------------------------------------

    # --- Pre-run Checks and Environment Setup ---
    # Set all provided credentials in the environment for LiteLLM to use.
    if GEMINI_API_KEY and "PASTE" not in GEMINI_API_KEY:
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

    if OPENAI_API_KEY and "PASTE" not in OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    if AWS_ACCESS_KEY_ID and "PASTE" not in AWS_ACCESS_KEY_ID:
        os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID

    if AWS_SECRET_ACCESS_KEY and "PASTE" not in AWS_SECRET_ACCESS_KEY:
        os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

    if AWS_REGION_NAME:
        os.environ["AWS_REGION_NAME"] = AWS_REGION_NAME

    # Verify that credentials for the selected models are available
    def check_creds_for_model(model_name):
        if not model_name: return True
        if "gemini" in model_name and not os.getenv("GEMINI_API_KEY"):
            print(f"\nERROR: Model '{model_name}' requires GEMINI_API_KEY, but it is not set.")
            return False
        if "openai" in model_name and not os.getenv("OPENAI_API_KEY"):
            print(f"\nERROR: Model '{model_name}' requires OPENAI_API_KEY, but it is not set.")
            return False
        if "bedrock" in model_name and not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
            print(f"\nERROR: Model '{model_name}' requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, but they are not set.")
            return False
        return True

    if not check_creds_for_model(PLANNER_MODEL) or not check_creds_for_model(HELPER_MODEL):
        print(f"Please edit 'run_planner.py' and set the required API key variables.")
        return

    # Clean up previous run if it exists
    if os.path.exists(output_dir):
        print(f"Removing previous output directory: {output_dir}")
        shutil.rmtree(output_dir)

    print(f"\nTopic: {topic}")

    # --- 3. Video Generator Initialization ---
    print("\nInitializing EnhancedVideoGenerator...")
    try:
        # Create a complete configuration object from the hardcoded values
        video_gen_config = VideoGenerationConfig(
            planner_model=PLANNER_MODEL,
            scene_model=HELPER_MODEL,
            helper_model=HELPER_MODEL,
            temperature=0.7,
            output_dir=output_dir,
            verbose=True, # Enable verbose for detailed test output
            use_rag=False, # Keep false for simple, repeatable tests
            use_context_learning=False,
            use_langfuse=True, # Enable for tracing
            enable_caching=False # Disable caching for a fresh run
        )

        # Instantiate the main generator class
        video_generator = EnhancedVideoGenerator(video_gen_config)

    except Exception as e:
        print(f"\nERROR: Failed to initialize EnhancedVideoGenerator: {e}")
        traceback.print_exc()
        return

    print("Initialization complete.")

    # --- 4. Execution ---
    print("\n--- Running Full Video Generation Pipeline ---")
    try:
        await video_generator.generate_video_pipeline(
            topic=topic,
            description=description,
            only_plan=False # Set to True to only generate plans without rendering
        )
    except Exception as e:
        print(f"\nERROR: An error occurred during the video generation pipeline: {e}")
        traceback.print_exc()
        return

    print(f"\n--- Live Video Generation Test Finished ---")
    print(f"Output files have been generated in the '{output_dir}/' directory.")

if __name__ == "__main__":
    # To run this script, edit the configuration section in the code
    # to set your API keys and desired models. Then run from your terminal:
    # python run_planner.py
    asyncio.run(main())
