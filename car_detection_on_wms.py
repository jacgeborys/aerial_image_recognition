import os
from _script.detector import CarDetector


def main():
    """Main execution with error handling"""
    try:
        # Set up base directory
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Simplified configuration without worker settings
        config = {
            'batch_size': 64,  # Reduced batch size
            'max_gpu_memory': 5.0,
            'queue_size': 128
        }

        # Initialize and run detector
        detector = CarDetector(base_dir, config)
        results = detector.detect(interactive=False)

        if results is not None and not results.empty:
            print("\nDetection completed successfully!")
            print(f"Results saved to: {detector.output_path}")
            return results
        else:
            print("\nNo results generated")
            return None

    except Exception as e:
        print(f"Error in main process: {str(e)}")
        return None

if __name__ == "__main__":
    main()