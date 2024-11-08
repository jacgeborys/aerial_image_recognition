import os
from _script.detector import CarDetector


def main():
    """Main execution with error handling"""
    try:
        # Set up base directory
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Custom configuration (optional)
        config = {
            'batch_size': 64,
            'num_workers': 25,
            'max_gpu_memory': 5.0,
            'queue_size': 64
        }

        # Initialize and run detector
        detector = CarDetector(base_dir, config)
        results = detector.detect(interactive=False)

        if results is not None and len(results) > 0:
            print("\nDetection completed successfully!")
            print(f"Results saved to: {os.path.join(base_dir, 'output', 'detections_results.geojson')}")
            return results
        else:
            print("\nNo results generated")
            return None

    except Exception as e:
        print(f"Error in main process: {str(e)}")
        return None

if __name__ == "__main__":
    main()