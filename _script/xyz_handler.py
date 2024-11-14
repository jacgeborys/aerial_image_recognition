import numpy as np
from PIL import Image, ImageEnhance
from io import BytesIO
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import mercantile
import concurrent.futures
from tqdm import tqdm
import traceback
import math
from shapely.geometry import Point
from pyproj import Transformer
import json
from datetime import datetime

class XYZHandler:
    def __init__(self, xyz_url, timeout=10, num_workers=25):
        self.xyz_url = xyz_url
        self.timeout = timeout
        self.num_workers = num_workers
        self.zoom = 21
        self.tile_size = 256
        
        # At zoom 21:
        # Each tile is ~19m at equator
        # 4x4 grid = ~76m
        # Crop to center 64m
        self.target_size = 64  # meters
        self.crop_size = 864   # pixels
        self.meters_per_pixel = 0.074  # at zoom 21
        
        # Configure requests session
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
        self.session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    
    def _fetch_tile(self, x, y, z):
        """Fetch single XYZ tile"""
        url = self.xyz_url.format(x=x, y=y, z=z)
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Error fetching tile {x},{y}: {str(e)}")
            return None

    def _fetch_surrounding_tiles(self, center_tile):
        """Fetch 4x4 grid of tiles"""
        tiles_to_fetch = []
        for dy in range(-1, 3):  # -1 to 2 (4 tiles)
            for dx in range(-1, 3):  # -1 to 2 (4 tiles)
                tile_x = center_tile.x + dx
                tile_y = center_tile.y + dy
                tiles_to_fetch.append((tile_x, tile_y, self.zoom))
        
        print("\n=== Tile Grid ===")
        print(f"Center tile: {center_tile.x},{center_tile.y}")
        print(f"Fetching 4x4 grid ({len(tiles_to_fetch)} tiles)")
        print(f"Expected merged size: 1024x1024px")
        
        # Create merged image
        merged_img = Image.new('RGB', (1024, 1024))
        
        # Track progress
        fetched = 0
        total = len(tiles_to_fetch)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_tile = {
                executor.submit(self._fetch_tile, x, y, z): (x, y)
                for x, y, z in tiles_to_fetch
            }
            
            for future in concurrent.futures.as_completed(future_to_tile):
                x, y = future_to_tile[future]
                try:
                    img = future.result()
                    if img:
                        # Calculate position in merged image
                        px = (x - (center_tile.x - 1)) * 256
                        py = (y - (center_tile.y - 1)) * 256
                        merged_img.paste(img, (px, py))
                        fetched += 1
                        print(f"\rDownloaded: {fetched}/{total} ({fetched/total*100:.1f}%)", end="")
                except Exception as e:
                    print(f"\nError fetching tile {x},{y}: {str(e)}")
        
        print(f"\nMerged Result:")
        print(f"Size: {merged_img.width}x{merged_img.height}px")
        print(f"Coverage: {merged_img.width * self.meters_per_pixel:.1f}m")
        
        return merged_img

    def get_single_image(self, bbox, max_retries=3):
        """Fetch single image and save tile boundary"""
        try:
            # Calculate center and get tile info
            center_lon = (bbox[0] + bbox[2])/2
            center_lat = (bbox[1] + bbox[3])/2
            center_tile = mercantile.tile(center_lon, center_lat, self.zoom)
            
            print("\n=== Tile Processing Debug ===")
            print(f"Input bbox: {bbox}")
            print(f"Center point: ({center_lon:.6f}, {center_lat:.6f})")
            print(f"Zoom level: {self.zoom}")
            print(f"Center tile coordinates: {center_tile.x}, {center_tile.y}, {center_tile.z}")
            
            # Save tile boundary to GeoJSON
            tile_geojson = {
                "type": "FeatureCollection",
                "features": [{
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
                    "properties": {
                        "type": "tile_boundary",
                        "zoom": self.zoom,
                        "center_tile": f"{center_tile.x},{center_tile.y}",
                        "coverage_meters": 64.0,
                        "pixel_size": 864,
                        "bbox": bbox
                    }
                }]
            }
            
            # Save to preview_tile.geojson
            with open('preview_tile.geojson', 'w') as f:
                json.dump(tile_geojson, f, indent=2)
            
            print("\nSaved tile boundary to preview_tile.geojson")
            
            # Calculate coverage
            lon_span = bbox[2] - bbox[0]
            lat_span = bbox[3] - bbox[1]
            meters_lon = lon_span * 111319 * math.cos(math.radians(center_lat))
            meters_lat = lat_span * 111319
            
            print(f"\n=== Coverage Analysis ===")
            print(f"Tile spans:")
            print(f"  Longitude: {lon_span:.6f}° ({meters_lon:.1f}m)")
            print(f"  Latitude:  {lat_span:.6f}° ({meters_lat:.1f}m)")
            print(f"Pixel resolution: {64/864:.4f}m/px")
            
            # Continue with image fetching...
            merged_img = self._fetch_surrounding_tiles(center_tile)
            if merged_img is None:
                return None
            
            # Crop to target area
            offset_x = (1024 - 864) // 2
            offset_y = (1024 - 864) // 2
            cropped = merged_img.crop((offset_x, offset_y, offset_x+864, offset_y+864))
            
            return [(cropped, bbox, None)]
            
        except Exception as e:
            print(f"Error in get_single_image: {str(e)}")
            traceback.print_exc()
            return None

    def _enhance_shadows(self, img):
        """Enhanced shadow processing"""
        # Increase brightness in darker areas while preserving detail
        enhancer = ImageEnhance.Brightness(img)
        brightened = enhancer.enhance(1.8)
        
        # Increase contrast to maintain definition
        contrast = ImageEnhance.Contrast(brightened)
        return contrast.enhance(1.2)

    def _crop_to_target_area(self, merged_img, bbox):
        """Crop with detailed coordinate tracking"""
        print("\n=== Coordinate Transform Debug ===")
        print(f"Input bbox: {bbox}")
        print(f"1. Original Image:")
        print(f"   Size: {merged_img.width}x{merged_img.height}px")
        print(f"   Coverage: {merged_img.width * self.meters_per_pixel:.1f}m x {merged_img.height * self.meters_per_pixel:.1f}m")
        
        # Calculate crop size
        target_pixels = int(64 / self.meters_per_pixel)
        center_x = merged_img.width // 2
        center_y = merged_img.height // 2
        half_size = target_pixels // 2
        
        # Calculate crop coordinates
        crop_box = (
            center_x - half_size,
            center_y - half_size,
            center_x + half_size,
            center_y + half_size
        )
        
        print(f"2. Crop Window:")
        print(f"   Pixels: {crop_box}")
        print(f"   Center: ({center_x}, {center_y})")
        print(f"   Size: {target_pixels}x{target_pixels}px")
        
        cropped = merged_img.crop(crop_box)
        final = cropped.resize((640, 640), Image.Resampling.LANCZOS)
        
        # Calculate coordinate transforms
        px_per_degree_lon = target_pixels / (bbox[2] - bbox[0])
        px_per_degree_lat = target_pixels / (bbox[3] - bbox[1])
        
        print(f"3. Coordinate Scaling:")
        print(f"   Longitude: {px_per_degree_lon:.1f}px/degree")
        print(f"   Latitude: {px_per_degree_lat:.1f}px/degree")
        print(f"   Meters/pixel: {self.meters_per_pixel:.3f}m")
        
        return final

    def fetch_batch(self, tile_batch, progress_bar=None):
        """Fetch multiple tiles in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for tile in tile_batch:
                future = executor.submit(self.get_single_image, tile)
                futures.append(future)
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    if progress_bar:
                        progress_bar.update(1)
                except Exception as e:
                    print(f"Error in fetch_batch: {str(e)}")
                    continue
            
            return results

    def nms_geographic(self, detections, distance_threshold=2):  # 2 meters
        """Non-maximum suppression based on geographic distance"""
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

    def _calculate_tile_size(self, center_lat):
        """Calculate tile size in meters at given latitude"""
        # Earth's circumference at equator
        earth_circumference = 40075016.686  # meters
        
        # Tile width in meters at this latitude
        tile_width_equator = earth_circumference / (2 ** self.zoom)
        tile_width = tile_width_equator * math.cos(math.radians(center_lat))
        
        print("\n=== Tile Size Verification ===")
        print(f"At latitude {center_lat:.6f}:")
        print(f"Single tile width: {tile_width:.1f}m")
        print(f"4x4 grid coverage: {tile_width * 4:.1f}m")
        print(f"Target coverage: 64.0m")
        
        return tile_width