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
from collections import OrderedDict


class SimpleDetector:
    def __init__(self, model_path, output_dir):
        self.zoom = 21
        self.model_size = 640
        self.confidence_threshold = 0.3
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        earth_circumference = 40075016.686
        self.meters_per_pixel = earth_circumference / (2**self.zoom) / 256
        
        # Try to use CUDA if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        print(f"Using ONNX providers: {providers}")
        
        self.model = ort.InferenceSession(
            model_path, 
            providers=providers
        )
        
        # Single session with adaptive delay
        self.session = requests.Session()
        self.session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
        
        self.xyz_url = 'http://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
        self.last_request_time = time.time()
        self.current_delay = 0.01
        self.consecutive_errors = 0
        self.session_idx = 0
        self.tile_cache = OrderedDict()
        self.small_tile_cache = {}
        self.tile_cache_size = 10000
        self.current_row_cache = {}  # Cache for current processing row
        self.previous_row_cache = {}  # Cache for previous row

    def _manage_cache(self, new_y):
        """Manage row-based caching when moving to new row"""
        if self.current_row_y != new_y:
            # Moving to new row
            self.previous_row_cache = self.current_row_cache
            self.current_row_cache = {}
            self.current_row_y = new_y
            
            # Clear previous row cache if moving up more than one row
            if abs(new_y - self.current_row_y) > 1:
                self.previous_row_cache = {}

    def _fetch_tile(self, x, y, z):
        """Enhanced tile fetching with row-based caching"""
        tile_key = (x, y, z)
        
        # Check row caches first
        if tile_key in self.current_row_cache:
            return self.current_row_cache[tile_key]
        if tile_key in self.previous_row_cache:
            return self.previous_row_cache[tile_key]
            
        # Then check main cache
        if tile_key in self.tile_cache:
            return self.tile_cache[tile_key]
        
        # Fetch new tile
        server = self.session_idx % 4
        self.session_idx = (self.session_idx + 1) % 4
        url = self.xyz_url.format(s=server, x=x, y=y, z=z)

        try:
            response = self.session.get(url, timeout=5)
            if response.status_code != 200:
                raise requests.exceptions.RequestException(f"Status code: {response.status_code}")
            
            tile_image = Image.open(BytesIO(response.content))
            
            # Store in both caches
            self.current_row_cache[tile_key] = tile_image
            
            # Store in main cache with size limit
            if len(self.tile_cache) >= self.tile_cache_size:
                self.tile_cache.popitem(last=False)
            self.tile_cache[tile_key] = tile_image
            
            return tile_image
        except requests.RequestException as e:
            tqdm.write(f"Error fetching tile {x}, {y}, {z}: {e}")
            return None
        
    def _calculate_small_tile_bounds(self, lat, lon, overlap=0.1):
        """Calculate bounds for small tiles to cover a big tile with overlap around a center point."""
        # Calculate the necessary pixels and degrees for the overlap
        meters_per_pixel = self.meters_per_pixel * math.cos(math.radians(lat))
        target_size_meters = self.model_size * meters_per_pixel * (1 + overlap)
        
        # Define bounds in meters around the central point
        meters_to_lon = 1.0 / (111319.9 * math.cos(math.radians(lat)))
        meters_to_lat = 1.0 / 111319.9
        half_size = target_size_meters / 2

        target_bounds = {
            'west': lon - half_size * meters_to_lon,
            'east': lon + half_size * meters_to_lon,
            'south': lat - half_size * meters_to_lat,
            'north': lat + half_size * meters_to_lat
        }
        
        # Calculate the tile range in XYZ format at the current zoom level
        nw_tile = mercantile.tile(target_bounds['west'], target_bounds['north'], self.zoom)
        se_tile = mercantile.tile(target_bounds['east'], target_bounds['south'], self.zoom)
        
        min_x = min(nw_tile.x, se_tile.x) - 1
        max_x = max(nw_tile.x, se_tile.x) + 1
        min_y = min(nw_tile.y, se_tile.y) - 1
        max_y = max(nw_tile.y, se_tile.y) + 1
        
        return min_x, min_y, max_x, max_y

        
    def _fetch_small_tile(self, x, y, z):
        """Fetch or reuse small XYZ tile from the cache."""
        tile_key = (x, y, z)
        
        # Check if the small tile is already in cache
        if tile_key in self.small_tile_cache:
            return self.small_tile_cache[tile_key]
        
        # Fetch from server if not cached
        server = self.session_idx % 4
        self.session_idx = (self.session_idx + 1) % 4
        url = self.xyz_url.format(s=server, x=x, y=y, z=z)

        try:
            response = self.session.get(url, timeout=5)
            if response.status_code != 200:
                raise requests.exceptions.RequestException(f"Status code: {response.status_code}")
            
            tile_image = Image.open(BytesIO(response.content))
            self.small_tile_cache[tile_key] = tile_image  # Cache the small tile
            return tile_image
        except requests.RequestException as e:
            tqdm.write(f"Error fetching tile {x}, {y}, {z}: {e}")
            return None

    def get_big_tile(self, lat, lon, overlap=0.1):
        """Generate a big tile with specified overlap by assembling small cached tiles."""
        # Calculate required pixels with overlap
        target_size_pixels = int(self.model_size * (1 + overlap))
        
        # Determine coordinates of small tiles needed to cover this big tile
        min_x, min_y, max_x, max_y = self._calculate_small_tile_bounds(lat, lon, overlap)
        
        # Assemble the big tile using cached small tiles
        big_tile = Image.new('RGB', (target_size_pixels, target_size_pixels))
        
        for ty in range(min_y, max_y + 1):
            for tx in range(min_x, max_x + 1):
                small_tile = self._fetch_small_tile(tx, ty, self.zoom)
                
                if small_tile:
                    # Calculate position within big tile and paste the small tile
                    px = (tx - min_x) * 256
                    py = (ty - min_y) * 256
                    big_tile.paste(small_tile, (px, py))
        
        # Crop the big tile to desired model input size (640x640 or as specified)
        cropped_big_tile = big_tile.crop((0, 0, self.model_size, self.model_size))
        return cropped_big_tile

    
    def get_image(self, lat, lon, target_size_meters=64):
        """Get image centered on coordinates with detailed tile fetching statistics"""
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
        
        # Initialize tile fetching statistics
        tiles_stats = {
            'total_tiles': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'total_fetch_time': 0,
            'tiles_data': []
        }
        
        fetch_start = time.time()
        total_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)
        tqdm.write(f"\nFetching {total_tiles} tiles for location {lat:.6f}, {lon:.6f}")
        
        for ty in range(min_y, max_y + 1):
            for tx in range(min_x, max_x + 1):
                tiles_stats['total_tiles'] += 1
                tile_start = time.time()
                
                img = self._fetch_tile(tx, ty, self.zoom)
                tile_time = time.time() - tile_start
                tiles_stats['total_fetch_time'] += tile_time
                
                tile_info = {
                    'tile_x': tx,
                    'tile_y': ty,
                    'fetch_time': tile_time,
                    'success': img is not None
                }
                tiles_stats['tiles_data'].append(tile_info)
                
                if img:
                    tiles_stats['successful_fetches'] += 1
                    px = (tx - min_x) * 256
                    py = (ty - min_y) * 256
                    merged.paste(img, (px, py))
                    
                    tile_bounds = mercantile.bounds(tx, ty, self.zoom)
                    tiles_info.append({
                        'tile_x': tx,
                        'tile_y': ty,
                        'zoom': self.zoom,
                        'position': (px, py),
                        'fetch_time': tile_time,
                        'bounds': {
                            'west': tile_bounds.west,
                            'east': tile_bounds.east,
                            'south': tile_bounds.south,
                            'north': tile_bounds.north
                        }
                    })
                else:
                    tiles_stats['failed_fetches'] += 1
        
        total_time = time.time() - fetch_start
        
        # Print tile fetching statistics
        tqdm.write("\nTile fetching summary:")
        tqdm.write(f"Total tiles: {tiles_stats['total_tiles']}")
        tqdm.write(f"Successful: {tiles_stats['successful_fetches']}")
        tqdm.write(f"Failed: {tiles_stats['failed_fetches']}")
        tqdm.write(f"Total time: {total_time:.2f}s")
        tqdm.write(f"Average time per tile: {tiles_stats['total_fetch_time']/tiles_stats['total_tiles']:.2f}s")
        
        # Calculate bounds and crop image
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
            },
            "tiles_stats": tiles_stats
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

    def _process_detections(self, boxes, lat, lon):
        """Process model detections to convert YOLO-relative coordinates to latitude and longitude."""
        detections = []
        
        # Calculate bounds using meters_per_pixel from the actual zoom level
        meters_per_pixel = self.meters_per_pixel * math.cos(math.radians(lat))
        half_size_meters = (self.model_size * meters_per_pixel) / 2
        
        # Convert half_size from meters to degrees more accurately
        half_size_lon = half_size_meters / (111319.9 * math.cos(math.radians(lat)))
        half_size_lat = half_size_meters / 111319.9
        
        bounds = {
            'west': lon - half_size_lon,
            'east': lon + half_size_lon,
            'south': lat - half_size_lat,
            'north': lat + half_size_lat
        }
        
        for box in boxes:
            x_yolo, y_yolo, w, h, conf = box[:5]
            
            if conf < self.confidence_threshold:
                continue

            # Convert YOLO coordinates to image fractions
            x_frac = x_yolo / self.model_size
            y_frac = y_yolo / self.model_size
            
            # Convert to lat/lon accounting for Mercator projection distortion
            lon_det = bounds['west'] + x_frac * (bounds['east'] - bounds['west'])
            lat_det = bounds['north'] - y_frac * (bounds['north'] - bounds['south'])

            detections.append({
                'lon': float(lon_det),
                'lat': float(lat_det),
                'confidence': float(conf),
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
        """Process a batch of points with timing analysis and correct coordinate conversion"""
        batch_detections = []
        batch_coverages = []
        
        timing_stats = {
            'tile_fetching': 0,
            'inference': 0,
            'coordinate_processing': 0
        }
        
        for lat, lon in points:
            try:
                # Get image and preview info for coordinate conversion
                t0 = time.time()
                image, preview_info, target_bounds = self.get_image(lat, lon)
                timing_stats['tile_fetching'] += time.time() - t0
                
                if image is not None:
                    # Run detection with correct coordinate conversion
                    t0 = time.time()
                    detections = self.detect(image, preview_info)
                    timing_stats['inference'] += time.time() - t0
                    
                    batch_detections.extend(detections)
                    
                    # Record coverage using the same bounds from preview_info
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
                continue
        
        return batch_detections, batch_coverages, timing_stats


if __name__ == "__main__":
    import time
    start_time = time.time()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'car_aerial_detection_yolo7_ITCVD_deepness.onnx')
    
    # Read the shapefile and set up output directory
    shp_path = os.path.join(base_dir, 'gis', 'frames', 'la.shp')
    frame_name = os.path.splitext(os.path.basename(shp_path))[0]
    output_dir = os.path.join(base_dir, 'output', frame_name)
    
    os.makedirs(output_dir, exist_ok=True)
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
                "total_detections": len(detections),
                "processing_time": time.time() - start_time
            }
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
    
    # Initialize timing dictionary
    timing = {
        'setup': 0,
        'grid_creation': 0,
        'processing': 0,
        'duplicate_removal': 0,
        'saving': 0
    }
    
    # Read shapefile and prepare points
    t0 = time.time()
    print("Reading shapefile and calculating grid...")
    
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
    
    points = []
    for lat in lats:
        for lon in lons:
            point = Point(lon, lat)
            if any(gdf.contains(point)):
                points.append((lat, lon))
    
    timing['grid_creation'] = time.time() - t0
    print(f"Grid creation took {timing['grid_creation']:.2f}s")
    print(f"Total points to process: {len(points)}")
    
    # Initialize detector
    t0 = time.time()
    detector = SimpleDetector(model_path, output_dir)
    timing['setup'] = time.time() - t0
    
    # Process points in batches
    batch_size = 100
    all_detections = []
    all_coverages = []
    processed_tiles = 0
    
    n_batches = math.ceil(len(points) / batch_size)
    batch_pbar = tqdm(total=n_batches, desc="Processing batches", unit="batch")
    
    t0 = time.time()
    try:
        for i in range(0, len(points), batch_size):
            batch_start = time.time()
            
            batch_points = points[i:i+batch_size]
            batch_detections, batch_coverages, batch_timing = detector.process_batch(batch_points)
            
            batch_time = time.time() - batch_start
            
            all_detections.extend(batch_detections)
            all_coverages.extend(batch_coverages)
            
            processed_tiles += len(batch_points)
            
            # Print batch timing every 5 batches or when there's an interesting event
            if i % (5 * batch_size) == 0 or batch_time > 30:  # Print if batch took more than 30s
                tqdm.write(f"\nBatch {i//batch_size + 1}/{n_batches}:")
                tqdm.write(f"  Tile fetching: {batch_timing['tile_fetching']:.2f}s")
                tqdm.write(f"  Model inference: {batch_timing['inference']:.2f}s")
                tqdm.write(f"  Coordinate processing: {batch_timing['coordinate_processing']:.2f}s")
                tqdm.write(f"  Total batch time: {batch_time:.2f}s")
                tqdm.write(f"  Cars detected: {len(batch_detections)}")
            
            # Save checkpoint every 2000 tiles
            if processed_tiles % 2000 < batch_size:
                tqdm.write(f"\nSaving checkpoint at {processed_tiles} tiles...")
                save_checkpoint(all_detections, all_coverages, frame_name, checkpoint_path)
            
            batch_pbar.update(1)
            
            # Dynamic delay based on batch time
            if batch_time < 20:  # If batch is processing quickly
                time.sleep(0.001)
        
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

    timing['processing'] = time.time() - t0
    
    # Remove duplicates
    print("\nRemoving duplicates...")
    t0 = time.time()
    all_detections = detector._remove_duplicates(all_detections, distance_threshold=1.0)
    timing['duplicate_removal'] = time.time() - t0
    
    # Save final results
    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine UTM zone for metadata
    utm_zone = int((bounds[0] + 180) / 6) + 1
    north = bounds[1] > 0
    epsg = f"326{utm_zone:02d}" if north else f"327{utm_zone:02d}"
    
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
            "processing_time": time.time() - start_time,
            "timing_breakdown": timing,
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
                "total_tiles": len(all_coverages),
                "processing_time": time.time() - start_time
            }
        }, f, indent=2)
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        
    timing['saving'] = time.time() - t0
    total_time = time.time() - start_time
    
    print(f"\nProcessing complete!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Timing breakdown:")
    print(f"  Setup: {timing['setup']:.2f}s")
    print(f"  Grid creation: {timing['grid_creation']:.2f}s")
    print(f"  Processing: {timing['processing']:.2f}s")
    print(f"  Duplicate removal: {timing['duplicate_removal']:.2f}s")
    print(f"  Saving: {timing['saving']:.2f}s")
    print(f"Total cars detected: {len(all_detections)}")
    print(f"Results saved to: {output_dir}")