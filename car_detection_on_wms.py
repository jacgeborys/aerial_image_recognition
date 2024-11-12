import os
from _script.detector import CarDetector

def main():
    """Main execution with error handling"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Only override what's different from defaults
        custom_config = {
            'frame_path': 'amsterdam.shp',  # Only specify what's different
            'wms_layer': 'Actueel_orthoHR'  # from DEFAULT_CONFIG
        }

        detector = CarDetector(base_dir, custom_config)
        results = detector.detect(interactive=False)
        
        if results is not None and len(results) > 0:
            print("\nDetection completed successfully!")
            print(f"Results saved to: {detector.output_dir}")
            return results
        else:
            print("\nNo results generated")
            return None

    except Exception as e:
        print(f"Error in main process: {str(e)}")
        return None

if __name__ == "__main__":
    main()