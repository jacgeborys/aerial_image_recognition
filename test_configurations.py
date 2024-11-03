import os
from _script.detector import CarDetector
from tqdm import tqdm

def test_configuration(base_dir, config, test_area_bbox=None):
    """Run detection with specific configuration and return results count"""
    try:
        detector = CarDetector(base_dir, config)
        results = detector.detect(interactive=False)
        if results is not None and not results.empty:
            return len(results)
        return 0
    except Exception as e:
        print(f"Error testing configuration: {str(e)}")
        return 0

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define test configurations
    configurations = [
        {
            'name': '100.0m → 15.6cm/px',
            'config': {
                'tile_size_meters': 100.0,
                'batch_size': 512,
                'num_workers': 32,
                'max_gpu_memory': 5.0,
                'queue_size': 512
            }
        },
        {
            'name': '64.0m → 10.0cm/px',
            'config': {
                'tile_size_meters': 64.0,
                'batch_size': 512,
                'num_workers': 32,
                'max_gpu_memory': 5.0,
                'queue_size': 512
            }
        },
        {
            'name': '51.2m → 8.0cm/px',
            'config': {
                'tile_size_meters': 51.2,
                'batch_size': 512,
                'num_workers': 32,
                'max_gpu_memory': 5.0,
                'queue_size': 512
            }
        },
        {
            'name': '25.0m → 3.9cm/px',
            'config': {
                'tile_size_meters': 25.0,
                'batch_size': 512,
                'num_workers': 32,
                'max_gpu_memory': 5.0,
                'queue_size': 512
            }
        },
        {
            'name': '20.0m → 3.1cm/px',
            'config': {
                'tile_size_meters': 20.0,
                'batch_size': 512,
                'num_workers': 32,
                'max_gpu_memory': 5.0,
                'queue_size': 512
            }
        }
    ]

    # Test each configuration
    results = []
    for config in tqdm(configurations, desc="Testing configurations"):
        print(f"\nTesting configuration: {config['name']}")
        count = test_configuration(base_dir, config['config'])
        results.append({
            'name': config['name'],
            'detections': count
        })
        print(f"Found {count} detections")

    # Print summary
    print("\nResults Summary:")
    print("-" * 50)
    for result in results:
        print(f"{result['name']}: {result['detections']} detections")

if __name__ == "__main__":
    main() 