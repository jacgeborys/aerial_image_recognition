import mercantile
import requests
from PIL import Image
import numpy as np
import onnxruntime as ort
from io import BytesIO
import math
import os
import json
from datetime import datetime
from PIL import ImageDraw
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

class SimpleDetector:
    def __init__(self, model_path, output_dir):
        self.zoom = 21
        self.model_size = 640
        self.confidence_threshold = 0.3
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        earth_circumference = 40075016.686
        self.meters_per_pixel = earth_circumference / (2**self.zoom) / 256
        
        self.model = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        
        self.session = requests.Session()
        self.session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
        
        self.xyz_url = 'http://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'

    def _fetch_tile(self, x, y, z, retries=3):
        url = self.xyz_url.format(x=x, y=y, z=z)
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            if retries > 0:
                return self._fetch_tile(x, y, z, retries-1)
            return None

    def get_image(self, lat, lon, target_size_meters=64):
        meters_per_pixel = self.meters_per_pixel * math.cos(math.radians(lat))
        pixels_needed = int(target_size_meters / meters_per_pixel)
        
        meters_to_lon = 1.0 / (111319.9 * math.cos(math.radians(lat)))
        meters_to_lat = 1.0 / 111319.9
        half_size = target_size_meters / 2
        
        target_bounds = {
            'west': lon - half_size * meters_to_lon,
            'east': lon + half_size * meters_to_lon,
            'south': lat - half_size * meters_to_lat,
            'north': lat + half_size * meters_to_lat
        }
        
        nw_tile = mercantile.tile(target_bounds['west'], target_bounds['north'], self.zoom)
        se_tile = mercantile.tile(target_bounds['east'], target_bounds['south'], self.zoom)
        
        min_x = min(nw_tile.x, se_tile.x) - 1
        max_x = max(nw_tile.x, se_tile.x) + 1
        min_y = min(nw_tile.y, se_tile.y) - 1
        max_y = max(nw_tile.y, se_tile.y) + 1
        
        tile_width = max_x - min_x + 1
        tile_height = max_y - min_y + 1
        merged = Image.new('RGB', (tile_width * 256, tile_height * 256))
        tiles_info = []
        
        for ty in range(min_y, max_y + 1):
            for tx in range(min_x, max_x + 1):
                img = self._fetch_tile(tx, ty, self.zoom)
                if img:
                    px = (tx - min_x) * 256
                    py = (ty - min_y) * 256
                    merged.paste(img, (px, py))
                    
                    tile_bounds = mercantile.bounds(tx, ty, self.zoom)
                    tiles_info.append({
                        'tile_x': tx,
                        'tile_y': ty,
                        'zoom': self.zoom,
                        'position': (px, py),
                        'bounds': {
                            'west': tile_bounds.west,
                            'east': tile_bounds.east,
                            'south': tile_bounds.south,
                            'north': tile_bounds.north
                        }
                    })
        
        merged_bounds = {
            'west': mercantile.bounds(min_x, min_y, self.zoom).west,
            'east': mercantile.bounds(max_x, max_y, self.zoom).east,
            'south': mercantile.bounds(min_x, max_y, self.zoom).south,
            'north': mercantile.bounds(max_x, min_y, self.zoom).north
        }
        
        merged_width = merged.width
        merged_height = merged.height
        
        x_scale = merged_width / (merged_bounds['east'] - merged_bounds['west'])
        y_scale = merged_height / (merged_bounds['north'] - merged_bounds['south'])
        
        left = int((target_bounds['west'] - merged_bounds['west']) * x_scale)
        top = int((merged_bounds['north'] - target_bounds['north']) * y_scale)
        right = left + pixels_needed
        bottom = top + pixels_needed
        
        cropped = merged.crop((left, top, right, bottom))
        
        preview_info = {
            "timestamp": datetime.now().isoformat(),
            "spatial_info": {
                "center": {"lat": lat, "lon": lon},
                "bounds": target_bounds,
                "merged_bounds": merged_bounds,
                "zoom_level": self.zoom,
                "meters_per_pixel": meters_per_pixel,
                "target_size_meters": target_size_meters,
                "scales": {
                    "x": float(x_scale),
                    "y": float(y_scale)
                }
            },
            "image_info": {
                "merged_size": [merged_width, merged_height],
                "crop_size": pixels_needed,
                "crop_offset": [left, top],
                "final_size": [pixels_needed, pixels_needed]
            },
            "tiles": tiles_info
        }
        
        return cropped, preview_info, target_bounds

    def detect(self, image, preview_info):
        detections = []
        bounds = preview_info['spatial_info']['bounds']
        crop_size = preview_info['image_info']['crop_size']
        
        image_640 = image.resize((self.model_size, self.model_size))
        img_array = np.array(image_640)
        img_array = img_array.astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        outputs = self.model.run(None, {self.model.get_inputs()[0].name: img_array})
        boxes = outputs[0][0]
        conf_mask = boxes[:, 4] >= self.confidence_threshold
        boxes = boxes[conf_mask]
        
        for box in boxes:
            x_yolo, y_yolo, w, h, conf = box[:5]
            
            x_frac = x_yolo / self.model_size
            y_frac = y_yolo / self.model_size
            
            x_img = x_frac * crop_size
            y_img = y_frac * crop_size
            
            lon = bounds['west'] + x_frac * (bounds['east'] - bounds['west'])
            lat = bounds['north'] - y_frac * (bounds['north'] - bounds['south'])
            
            detections.append({
                'lon': float(lon),
                'lat': float(lat),
                'confidence': float(conf),
                'image': {'x': float(x_img), 'y': float(y_img)},
                'yolo': {'x': float(x_yolo), 'y': float(y_yolo)}
            })
        
        return self._remove_duplicates(detections, distance_threshold=0.5)

    def _remove_duplicates(self, detections, distance_threshold=1.0):
        """Remove duplicate detections within distance threshold in meters"""
        if not detections:
            return []
        
        # Sort by confidence to keep the highest confidence detection when duplicates are found
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        kept = []
        
        for det1 in detections:
            keep = True
            for det2 in kept:
                # Calculate distance in meters
                dx = (det1['lon'] - det2['lon']) * 111319.9 * math.cos(math.radians(det1['lat']))
                dy = (det1['lat'] - det2['lat']) * 111319.9
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < distance_threshold:
                    keep = False
                    break
            if keep:
                kept.append(det1)
        
        return kept

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'car_aerial_detection_yolo7_ITCVD_deepness.onnx')
    
    # Read the shapefile and set up output directory
    shp_path = os.path.join(base_dir, 'gis', 'frames', 'la_test.shp')
    frame_name = os.path.splitext(os.path.basename(shp_path))[0]
    output_dir = os.path.join(base_dir, 'output', frame_name)
    
    gdf = gpd.read_file(shp_path)
    bounds = gdf.total_bounds
    
    spacing_meters = 60
    lat_center = (bounds[1] + bounds[3]) / 2
    meters_to_lon = 1.0 / (111319.9 * math.cos(math.radians(lat_center)))
    meters_to_lat = 1.0 / 111319.9
    
    spacing_lon = spacing_meters * meters_to_lon
    spacing_lat = spacing_meters * meters_to_lat
    
    lons = np.arange(bounds[0], bounds[2], spacing_lon)
    lats = np.arange(bounds[1], bounds[3], spacing_lat)
    
    detector = SimpleDetector(model_path, output_dir)
    
    all_detections = []
    tile_coverages = []
    
    total_points = len(lons) * len(lats)
    progress_bar = tqdm(total=total_points, desc="Processing grid points")
    
    for lat in lats:
        for lon in lons:
            point = Point(lon, lat)
            
            if any(gdf.contains(point)):
                try:
                    image, preview_info, target_bounds = detector.get_image(lat, lon)
                    detections = detector.detect(image, preview_info)
                    
                    all_detections.extend(detections)
                    tile_coverages.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [target_bounds['west'], target_bounds['south']],
                                [target_bounds['east'], target_bounds['south']],
                                [target_bounds['east'], target_bounds['north']],
                                [target_bounds['west'], target_bounds['north']],
                                [target_bounds['west'], target_bounds['south']]
                            ]]
                        },
                        "properties": {
                            "center": {"lat": lat, "lon": lon},
                            "cars_detected": len(detections)
                        }
                    })
                    
                except Exception as e:
                    tqdm.write(f"Error at {lat:.6f}, {lon:.6f}: {str(e)}")
            
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Remove duplicates from the complete set of detections
    print("Removing duplicate detections...")
    all_detections = detector._remove_duplicates(all_detections, distance_threshold=1.0)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detections
    detections_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [det['lon'], det['lat']]
                },
                "properties": {
                    "confidence": det['confidence']
                }
            }
            for det in all_detections
        ],
        "metadata": {
            "timestamp": timestamp,
            "total_points_processed": total_points,
            "total_detections": len(all_detections),
            "duplicate_threshold_meters": 1.0
        }
    }
    
    # Save coverage tiles
    coverage_geojson = {
        "type": "FeatureCollection",
        "features": tile_coverages,
        "metadata": {
            "timestamp": timestamp,
            "total_tiles": len(tile_coverages)
        }
    }
    
    # Save both files
    with open(os.path.join(output_dir, f"{frame_name}_detections_{timestamp}.geojson"), 'w') as f:
        json.dump(detections_geojson, f, indent=2)
    
    with open(os.path.join(output_dir, f"{frame_name}_coverage_{timestamp}.geojson"), 'w') as f:
        json.dump(coverage_geojson, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Total cars detected: {len(all_detections)}")
    print(f"Results saved to: {output_dir}")