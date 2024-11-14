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
from tqdm.auto import tqdm
import time
from pyproj import Transformer
import sys

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
        
        # Single session with adaptive delay
        self.session = requests.Session()
        self.session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
        
        self.xyz_url = 'http://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
        self.last_request_time = time.time()
        self.current_delay = 0.01  # Start with minimal delay
        self.consecutive_errors = 0

    def _fetch_tile(self, x, y, z, retries=3):
        """Fetch single XYZ tile with adaptive delay"""
        # Use rotating servers but single session
        server = int(time.time() * 1000) % 4  # Simple rotation based on time
        
        # Minimal delay between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.current_delay:
            time.sleep(self.current_delay - time_since_last)
        
        url = self.xyz_url.format(s=server, x=x, y=y, z=z)
        
        try:
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            self.last_request_time = time.time()
            
            # Success - gradually decrease delay if it's high
            if self.current_delay > 0.01:
                self.current_delay = max(0.01, self.current_delay * 0.9)
            self.consecutive_errors = 0
            
            return Image.open(BytesIO(response.content))
            
        except Exception as e:
            self.consecutive_errors += 1
            
            # Exponentially increase delay after consecutive errors
            if self.consecutive_errors > 1:
                self.current_delay = min(2.0, self.current_delay * 2)
            
            if retries > 0:
                tqdm.write(f"Error fetching tile, increasing delay to {self.current_delay:.3f}s")
                time.sleep(self.current_delay)
                return self._fetch_tile(x, y, z, retries-1)
            return None
    
    def get_image(self, lat, lon, target_size_meters=64):
        """Get image centered on coordinates with correct bounds calculation"""
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
            }
        }
        
        return cropped, preview_info, target_bounds
    






    def detect(self, image, preview_info):
        """Run YOLO detection with correct coordinate alignment"""
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
        
        return detections

    def _remove_duplicates(self, detections, distance_threshold=1.0):
        """Remove duplicate detections within distance threshold using UTM coordinates"""
        if not detections:
            return []
        
        # Determine UTM zone from the first detection's longitude
        utm_zone = int((detections[0]['lon'] + 180) / 6) + 1
        north = detections[0]['lat'] > 0
        epsg = f"326{utm_zone:02d}" if north else f"327{utm_zone:02d}"
        
        # Create transformer from WGS84 to UTM
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        
        # Convert all detections to UTM
        utm_detections = []
        for det in detections:
            x, y = transformer.transform(det['lon'], det['lat'])
            utm_detections.append({
                'x': x,
                'y': y,
                'original': det
            })
        
        # Sort by confidence
        utm_detections.sort(key=lambda x: x['original']['confidence'], reverse=True)
        kept = []
        
        for det1 in utm_detections:
            keep = True
            for det2 in kept:
                dx = det1['x'] - det2['x']
                dy = det1['y'] - det2['y']
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < distance_threshold:
                    keep = False
                    break
            if keep:
                kept.append(det1)
        
        return [det['original'] for det in kept]

    def process_batch(self, points):
        """Process a batch of points with minimal delays"""
        batch_detections = []
        batch_coverages = []
        
        for lat, lon in points:
            try:
                image, preview_info, target_bounds = self.get_image(lat, lon)
                if image is not None:
                    detections = self.detect(image, preview_info)
                    
                    batch_detections.extend(detections)
                    batch_coverages.append({
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
                time.sleep(0.5)  # Brief pause after error
                continue
        
        return batch_detections, batch_coverages

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'car_aerial_detection_yolo7_ITCVD_deepness.onnx')
    
    # Read the shapefile and set up output directory
    shp_path = os.path.join(base_dir, 'gis', 'frames', 'la.shp')
    frame_name = os.path.splitext(os.path.basename(shp_path))[0]
    output_dir = os.path.join(base_dir, 'output', frame_name)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define checkpoint path
    checkpoint_path = os.path.join(output_dir, f'checkpoint_{frame_name}.geojson')
    
    def save_checkpoint(detections, coverages, frame_name, checkpoint_path):
        """Save current progress to checkpoint file"""
        checkpoint_data = {
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
                for det in detections
            ],
            "coverage": coverages,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "frame_name": frame_name,
                "processed_tiles": processed_tiles,
                "total_detections": len(detections)
            }
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
    
    # Read shapefile and prepare points
    gdf = gpd.read_file(shp_path)
    bounds = gdf.total_bounds
    
    # Calculate grid points
    spacing_meters = 60
    lat_center = (bounds[1] + bounds[3]) / 2
    meters_to_lon = 1.0 / (111319.9 * math.cos(math.radians(lat_center)))
    meters_to_lat = 1.0 / 111319.9
    
    spacing_lon = spacing_meters * meters_to_lon
    spacing_lat = spacing_meters * meters_to_lat
    
    lons = np.arange(bounds[0], bounds[2], spacing_lon)
    lats = np.arange(bounds[1], bounds[3], spacing_lat)
    
    # Create list of points within polygon
    points = []
    for lat in lats:
        for lon in lons:
            point = Point(lon, lat)
            if any(gdf.contains(point)):
                points.append((lat, lon))
    
    print(f"Total points to process: {len(points)}")
    
    # Initialize detector
    detector = SimpleDetector(model_path, output_dir)
    
    # Process points in batches
    batch_size = 25
    all_detections = []
    all_coverages = []
    processed_tiles = 0
    
    n_batches = math.ceil(len(points) / batch_size)
    batch_pbar = tqdm(total=n_batches, desc="Processing batches")
    
    try:
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            batch_detections, batch_coverages = detector.process_batch(batch_points)
            
            all_detections.extend(batch_detections)
            all_coverages.extend(batch_coverages)
            
            processed_tiles += len(batch_points)
            
            # Save checkpoint every 2000 tiles
            if processed_tiles % 2000 < batch_size:
                tqdm.write(f"\nSaving checkpoint at {processed_tiles} tiles...")
                save_checkpoint(all_detections, all_coverages, frame_name, checkpoint_path)
            
            batch_pbar.update(1)
            
            # Very small delay between batches if everything is going well
            if detector.consecutive_errors == 0:
                time.sleep(0.1)
            
            if processed_tiles % 100 == 0:
                tqdm.write(f"Current delay: {detector.current_delay:.3f}s")
        
        batch_pbar.close()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving checkpoint...")
        save_checkpoint(all_detections, all_coverages, frame_name, checkpoint_path)
        sys.exit(0)
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Saving checkpoint...")
        save_checkpoint(all_detections, all_coverages, frame_name, checkpoint_path)
        raise

    
    print("\nRemoving duplicates...")
    all_detections = detector._remove_duplicates(all_detections, distance_threshold=1.0)
    
    # Prepare final outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine UTM zone for metadata
    utm_zone = int((bounds[0] + 180) / 6) + 1
    north = bounds[1] > 0
    epsg = f"326{utm_zone:02d}" if north else f"327{utm_zone:02d}"
    
    # Save final results
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
            "total_detections": len(all_detections),
            "duplicate_removal": {
                "threshold_meters": 1.0,
                "coordinate_system": f"EPSG:{epsg}",
                "utm_zone": utm_zone
            }
        }
    }
    
    # Save final files
    with open(os.path.join(output_dir, f"{frame_name}_detections_{timestamp}.geojson"), 'w') as f:
        json.dump(detections_geojson, f, indent=2)
    
    with open(os.path.join(output_dir, f"{frame_name}_coverage_{timestamp}.geojson"), 'w') as f:
        json.dump({
            "type": "FeatureCollection",
            "features": all_coverages,
            "metadata": {
                "timestamp": timestamp,
                "total_tiles": len(all_coverages)
            }
        }, f, indent=2)
    
    # Clean up checkpoint file after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print(f"\nProcessing complete!")
    print(f"Total cars detected: {len(all_detections)}")
    print(f"Results saved to: {output_dir}")