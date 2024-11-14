import os
from _script.detector import CarDetector
import traceback

def main():
    """Main execution with error handling"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Update config for XYZ tiles
        custom_config = {
            'frame_path': 'la_test.shp',
            'xyz_url': 'http://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',  # Google Satellite
            'use_xyz': True  # Flag to use XYZ instead of WMS
        }

        detector = CarDetector(base_dir, custom_config)
        results = detector.detect(interactive=False, force_restart=True)
        
        if results is not None and len(results) > 0:
            print("\nDetection completed successfully!")
            print(f"Results saved to: {detector.output_dir}")
            return results
        else:
            print("\nNo results generated")
            return None

    except Exception as e:
        print(f"Error in main process: {str(e)}")
        traceback.print_exc()
        return None

def add_tile_boundary(bbox):
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [bbox[0], bbox[1]],  # min lon, min lat
                [bbox[2], bbox[1]],  # max lon, min lat
                [bbox[2], bbox[3]],  # max lon, max lat
                [bbox[0], bbox[3]],  # min lon, max lat
                [bbox[0], bbox[1]]   # close polygon
            ]]
        },
        "properties": {"type": "tile_boundary"}
    }

def nms_geographic(detections, distance_threshold=2):  # 2 meters
    """Non-maximum suppression based on geographic distance"""
    from shapely.geometry import Point
    from pyproj import Transformer
    
    # Convert to UTM for meter-based distance calculations
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True)
    
    points = []
    for d in detections:
        x, y = transformer.transform(d['lon'], d['lat'])
        points.append((Point(x, y), d['confidence'], d))
    
    # Sort by confidence
    points.sort(key=lambda x: x[1], reverse=True)
    
    kept = []
    for p1, conf1, det1 in points:
        keep = True
        for p2, _, _ in kept:
            if p1.distance(p2) < distance_threshold:
                keep = False
                break
        if keep:
            kept.append((p1, conf1, det1))
    
    return [det for _, _, det in kept]

if __name__ == "__main__":
    main()