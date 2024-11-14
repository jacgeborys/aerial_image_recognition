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

class SimpleDetector:
    def __init__(self, model_path, output_dir):
        self.zoom = 21
        self.model_size = 640
        self.confidence_threshold = 0.3
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate meters per pixel at zoom 21 (at equator)
        earth_circumference = 40075016.686  # meters
        self.meters_per_pixel = earth_circumference / (2**self.zoom) / 256
        print(f"Base resolution: {self.meters_per_pixel:.4f}m/pixel at equator")
        
        print(f"Loading model from: {model_path}")
        self.model = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        
        self.session = requests.Session()
        self.session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
        
        self.xyz_url = 'http://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'

    def _fetch_tile(self, x, y, z, retries=3):
        """Fetch single XYZ tile with retry logic"""
        url = self.xyz_url.format(x=x, y=y, z=z)
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Error fetching tile {x},{y}: {str(e)}")
            if retries > 0:
                print(f"Retrying... ({retries} attempts left)")
                return self._fetch_tile(x, y, z, retries-1)
            return None

    def get_image(self, lat, lon, target_size_meters=64):
        """Get image centered on coordinates with correct bounds calculation"""
        # Calculate pixels needed
        meters_per_pixel = self.meters_per_pixel * math.cos(math.radians(lat))
        pixels_needed = int(target_size_meters / meters_per_pixel)
        print(f"\nSpatial calculations:")
        print(f"Latitude: {lat:.6f}")
        print(f"Longitude: {lon:.6f}")
        print(f"Resolution at latitude: {meters_per_pixel:.4f}m/pixel")
        print(f"Pixels needed for {target_size_meters}m: {pixels_needed}")
        
        # Calculate the geographic bounds we need to cover
        meters_to_lon = 1.0 / (111319.9 * math.cos(math.radians(lat)))
        meters_to_lat = 1.0 / 111319.9
        half_size = target_size_meters / 2
        
        target_bounds = {
            'west': lon - half_size * meters_to_lon,
            'east': lon + half_size * meters_to_lon,
            'south': lat - half_size * meters_to_lat,
            'north': lat + half_size * meters_to_lat
        }
        
        print(f"\nTarget area bounds:")
        print(f"West:  {target_bounds['west']:.6f}")
        print(f"East:  {target_bounds['east']:.6f}")
        print(f"South: {target_bounds['south']:.6f}")
        print(f"North: {target_bounds['north']:.6f}")
        
        # Get tiles that cover this area
        nw_tile = mercantile.tile(target_bounds['west'], target_bounds['north'], self.zoom)
        se_tile = mercantile.tile(target_bounds['east'], target_bounds['south'], self.zoom)
        
        # Ensure we get enough tiles
        min_x = min(nw_tile.x, se_tile.x) - 1
        max_x = max(nw_tile.x, se_tile.x) + 1
        min_y = min(nw_tile.y, se_tile.y) - 1
        max_y = max(nw_tile.y, se_tile.y) + 1
        
        # Create merged image
        tile_width = max_x - min_x + 1
        tile_height = max_y - min_y + 1
        merged = Image.new('RGB', (tile_width * 256, tile_height * 256))
        tiles_info = []
        
        print("\nFetching tiles...")
        fetched_tiles = 0
        
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
                    fetched_tiles += 1
                    print(f"Progress: {fetched_tiles} tiles", end='\r')
        
        print(f"\nFetched {fetched_tiles} tiles")
        
        # Calculate precise pixel coordinates for our target area
        merged_bounds = {
            'west': mercantile.bounds(min_x, min_y, self.zoom).west,
            'east': mercantile.bounds(max_x, max_y, self.zoom).east,
            'south': mercantile.bounds(min_x, max_y, self.zoom).south,
            'north': mercantile.bounds(max_x, min_y, self.zoom).north
        }
        
        # Calculate crop parameters
        merged_width = merged.width
        merged_height = merged.height
        
        x_scale = merged_width / (merged_bounds['east'] - merged_bounds['west'])
        y_scale = merged_height / (merged_bounds['north'] - merged_bounds['south'])
        
        left = int((target_bounds['west'] - merged_bounds['west']) * x_scale)
        top = int((merged_bounds['north'] - target_bounds['north']) * y_scale)
        right = left + pixels_needed
        bottom = top + pixels_needed
        
        print(f"\nCrop parameters:")
        print(f"Left: {left}, Top: {top}, Width: {pixels_needed}, Height: {pixels_needed}")
        
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
        
        # Save files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preview_path = os.path.join(self.output_dir, f"preview_{timestamp}")
        
        merged.save(f"{preview_path}_full.jpg")
        cropped.save(f"{preview_path}_cropped.jpg")
        
        # Save boundary GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": [{
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
                "properties": preview_info
            }]
        }
        
        with open(f"{preview_path}_boundary.geojson", 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"\nSaved preview files to: {preview_path}")
        return cropped, preview_info

    def detect(self, image, preview_info):
        """Run YOLO detection with correct coordinate alignment"""
        detections = []
        bounds = preview_info['spatial_info']['bounds']
        crop_size = preview_info['image_info']['crop_size']
        
        # Prepare and run YOLO detection
        image_640 = image.resize((self.model_size, self.model_size))
        img_array = np.array(image_640)
        img_array = img_array.astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        outputs = self.model.run(None, {self.model.get_inputs()[0].name: img_array})
        boxes = outputs[0][0]
        conf_mask = boxes[:, 4] >= self.confidence_threshold
        boxes = boxes[conf_mask]
        
        # Convert detections to geographic coordinates
        for box in boxes:
            x_yolo, y_yolo, w, h, conf = box[:5]
            
            # Convert YOLO coordinates to image fractions
            x_frac = x_yolo / self.model_size
            y_frac = y_yolo / self.model_size
            
            # Convert to image coordinates
            x_img = x_frac * crop_size
            y_img = y_frac * crop_size
            
            # Convert to geographic coordinates
            lon = bounds['west'] + x_frac * (bounds['east'] - bounds['west'])
            lat = bounds['north'] - y_frac * (bounds['north'] - bounds['south'])
            
            detections.append({
                'lon': float(lon),
                'lat': float(lat),
                'confidence': float(conf),
                'image': {'x': float(x_img), 'y': float(y_img)},
                'yolo': {'x': float(x_yolo), 'y': float(y_yolo)}
            })
        
        # Remove duplicates
        unique_detections = self._remove_duplicates(detections, distance_threshold=0.5)
        
        # Create visualization
        debug_image = image.copy()
        draw = ImageDraw.Draw(debug_image)
        
        for det in unique_detections:
            x = det['image']['x']
            y = det['image']['y']
            
            r = 5
            draw.ellipse([x-r, y-r, x+r, y+r], fill='red')
            draw.text((x+r+2, y-r), f"{det['confidence']:.2f}", fill='red')
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [det['lon'], det['lat']]
                    },
                    "properties": {
                        "confidence": det['confidence'],
                        "yolo": det['yolo'],
                        "image": det['image'],
                        "detection_id": i
                    }
                }
                for i, det in enumerate(unique_detections)
            ],
            "bbox": [
                bounds['west'],
                bounds['south'],
                bounds['east'],
                bounds['north']
            ],
            "metadata": {
                "timestamp": timestamp,
                "image_size": [crop_size, crop_size],
                "model_size": self.model_size,
                "total_detections": len(unique_detections),
                "confidence_threshold": self.confidence_threshold
            }
        }
        
        geojson_path = os.path.join(self.output_dir, f"detections_{timestamp}.geojson")
        with open(geojson_path, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        
        debug_path = os.path.join(self.output_dir, f"detection_visualization_{timestamp}.jpg")
        debug_image.save(debug_path)
        
        return unique_detections

    def _remove_duplicates(self, detections, distance_threshold=0.5):
        """Remove duplicate detections within distance threshold (meters)"""
        if not detections:
            return []
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        kept = []
        for det1 in detections:
            keep = True
            for det2 in kept:
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
    output_dir = os.path.join(base_dir, 'output', 'la')
    
    detector = SimpleDetector(model_path, output_dir)
    
    # Los Angeles coordinates
    lat, lon = 34.010932, -118.318263
    
    # Get image and run detection
    image, preview_info = detector.get_image(lat, lon)
    detections = detector.detect(image, preview_info)
    
    print(f"\nFound {len(detections)} unique cars")
    print(f"Results saved to: {output_dir}")