from owslib.wms import WebMapService
from PIL import Image
from io import BytesIO
import time
import concurrent.futures
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import box
import os
import random


class WMSHandler:
    def __init__(self, wms_url, layer='Actueel_orthoHR', srs='EPSG:4326', size=(1280, 1280), 
                 image_format='image/jpeg', timeout=45, num_workers=25):
        """Initialize with WGS84 as default SRS"""
        self.wms_url = wms_url
        self.timeout = timeout
        self.num_workers = num_workers
        self.layer = layer
        self.srs = srs
        self.size = size
        self.image_format = image_format
        
        # Initialize failure tracking
        self.failed_tiles_log = []
        
        # Initialize other attributes
        self.wms = None
        self.session = self._create_session()
        self.stats = {
            'attempts': 0,
            'timeouts': 0,
            'timeout_attempts': 0,
            'successes': 0,
            'failures': 0,
            'total_bytes': 0,
            'start_time': time.time()
        }
        
        if not self._connect():
            raise RuntimeError("Failed to establish WMS connection")

    def _create_session(self):
        """Create session with optimized settings"""
        session = requests.Session()
        
        # More conservative retry strategy
        retry_strategy = Retry(
            total=5,  # Increased from 3
            backoff_factor=0.5,  # More aggressive backoff
            status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
            respect_retry_after_header=True,
            allowed_methods=["GET"],
            raise_on_status=False
        )
        
        # Configure connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=100,    # Increased pool size
            pool_maxsize=100,        # Increased max size
            pool_block=True          # Wait for connection when pool is full
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Add headers to look more like a regular client
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/jpeg,image/png,image/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
        
        return session

    def _connect(self):
        """Establish WMS connection"""
        try:
            self.wms = WebMapService(self.wms_url, version='1.3.0')
            return True
        except Exception as e:
            print(f"WMS connection error: {str(e)}")
            return False

    def _print_stats(self):
        """Print detailed WMS connection statistics"""
        if self.stats['attempts'] > 0:
            success_rate = (self.stats['successes'] / self.stats['attempts']) * 100
            stats_str = (
                f"\rWMS Stats: "
                f"{self.stats['successes']}/{self.stats['attempts']} OK ({success_rate:.1f}%) | "
                f"Timeouts: {self.stats['timeouts']} | "
                f"Timeout Attempts: {self.stats['timeout_attempts']} | "
                f"Speed: {self.stats['successes'] / (time.time() - self.stats['start_time']):.1f} img/s | "
                f"Avg Size: {self.stats.get('total_bytes', 0) / (self.stats['successes'] or 1) / 1024 / 1024:.1f}MB"
            )
            print(stats_str, end='', flush=True)

    def get_single_image(self, bbox, max_retries=5, initial_delay=0.1):
        """Fetch single image with failure logging"""
        self.stats['attempts'] += 1
        
        for attempt in range(max_retries):
            try:
                response = self.wms.getmap(
                    layers=[self.layer],
                    srs=self.srs,
                    bbox=bbox,
                    size=self.size,
                    format=self.image_format,
                    transparent=True
                )
                
                try:
                    img = Image.open(BytesIO(response.read()))
                    self.stats['successes'] += 1
                    return img
                except Exception as img_error:
                    print(f"\nError reading image from response: {str(img_error)}")
                    self.stats['failures'] += 1
                
            except Exception as e:
                if attempt == max_retries - 1:  # On final attempt
                    # Log the failed tile with details
                    self.failed_tiles_log.append({
                        'bbox': bbox,
                        'error': str(e),
                        'attempt': attempt + 1,
                        'timestamp': time.time()
                    })
                    
                if 'timeout' in str(e).lower():
                    self.stats['timeout_attempts'] += 1
                self.stats['failures'] += 1
                
                print(f"\nAttempt {attempt + 1}/{max_retries} failed for tile {bbox}")
                print(f"Error: {str(e)}")
                
                time.sleep(initial_delay * (2 ** attempt))
                continue
        
        self.stats['timeouts'] += 1
        return None

    def analyze_failures(self):
        """Analyze patterns in failed tiles"""
        if not self.failed_tiles_log:
            return
            
        print("\nFailure Analysis:")
        print("-" * 50)
        
        # Analyze bbox patterns
        bbox_sizes = []
        for fail in self.failed_tiles_log:
            bbox = fail['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            bbox_sizes.append((width, height))
        
        # Calculate statistics
        avg_width = sum(w for w, h in bbox_sizes) / len(bbox_sizes)
        avg_height = sum(h for w, h in bbox_sizes) / len(bbox_sizes)
        
        print(f"Total Failed Tiles: {len(self.failed_tiles_log)}")
        print(f"Average Failed Tile Size: {avg_width:.6f}° x {avg_height:.6f}°")
        
        # Check for patterns in errors
        error_types = {}
        for fail in self.failed_tiles_log:
            error_msg = fail['error']
            error_types[error_msg] = error_types.get(error_msg, 0) + 1
        
        print("\nError Types:")
        for error, count in error_types.items():
            print(f"- {error}: {count} occurrences")
            
        # Check for spatial patterns
        print("\nSpatial Distribution:")
        min_lon = min(fail['bbox'][0] for fail in self.failed_tiles_log)
        max_lon = max(fail['bbox'][2] for fail in self.failed_tiles_log)
        min_lat = min(fail['bbox'][1] for fail in self.failed_tiles_log)
        max_lat = max(fail['bbox'][3] for fail in self.failed_tiles_log)
        
        print(f"Failed tiles boundary box:")
        print(f"Longitude: {min_lon:.6f} to {max_lon:.6f}")
        print(f"Latitude: {min_lat:.6f} to {max_lat:.6f}")

    def fetch_batch(self, tiles, progress_bar=None):
        """Process batch with failure analysis"""
        if not tiles:
            return []
        
        print("\nStarting batch download...")
        self._print_stats()
        
        results = []
        failed_tiles = []
        
        # Reduce concurrent requests
        actual_workers = min(self.num_workers, 25)  # Cap at 25
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit with delay between submissions
            futures = []
            for bbox in tiles:
                time.sleep(0.05)  # Small delay between submissions
                futures.append((executor.submit(self.get_single_image, bbox), bbox))
            
            # Process results
            for future, bbox in futures:
                try:
                    img = future.result(timeout=self.timeout)
                    if img is not None:
                        results.append((img, bbox))
                    else:
                        failed_tiles.append(bbox)
                    if progress_bar:
                        progress_bar.update(1)
                except Exception as e:
                    self.stats['failures'] += 1
                    failed_tiles.append(bbox)
                    print(f"\nError for tile {bbox}: {str(e)}")
                
                self._print_stats()
        
        # Retry failed tiles with increased delays
        if failed_tiles:
            print(f"\nRetrying {len(failed_tiles)} failed tiles...")
            for bbox in failed_tiles:
                for delay in [2, 4, 8]:  # Increased delays
                    time.sleep(delay)
                    img = self.get_single_image(bbox, max_retries=3, initial_delay=1.0)
                    if img is not None:
                        results.append((img, bbox))
                        break
        
        # After processing all tiles, analyze failures
        if len(self.failed_tiles_log) > 0:
            self.analyze_failures()
        
        return results

    def fetch_all(self, all_tiles, batch_size=1024):
        """Fetch all tiles with batching and progress tracking"""
        total_batches = (len(all_tiles) + batch_size - 1) // batch_size
        all_results = []

        with tqdm(total=len(all_tiles), desc="Fetching tiles", unit="tiles") as pbar:
            for i in range(0, len(all_tiles), batch_size):
                batch = all_tiles[i:i + batch_size]
                batch_results = self.fetch_batch(batch, pbar)
                all_results.extend(batch_results)

        return all_results

    def preview_tiles(self, tiles, output_dir, prefix="tiles"):
        """Generate a GeoJSON preview of the tiles for QGIS visualization"""
        print(f"\nGenerating tile preview for {len(tiles)} tiles...")
        
        # Create geometries and calculate areas
        geometries = []
        areas = []
        
        # Create UTM transformer for area calculation
        mid_tile = tiles[len(tiles)//2]  # Get middle tile for UTM zone
        mid_lon = (mid_tile[0] + mid_tile[2]) / 2
        mid_lat = (mid_tile[1] + mid_tile[3]) / 2
        utm_zone = int((mid_lon + 180) / 6) + 1
        utm_epsg = f"326{utm_zone}" if mid_lat >= 0 else f"327{utm_zone}"
        
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
        
        for i, (minx, miny, maxx, maxy) in enumerate(tiles):
            # Create box geometry
            tile_box = box(minx, miny, maxx, maxy)
            geometries.append(tile_box)
            
            # Calculate area in square meters using UTM projection
            utm_coords = transformer.transform(
                [minx, maxx, maxx, minx],
                [miny, miny, maxy, maxy]
            )
            utm_box = box(
                utm_coords[0][0], utm_coords[1][0],
                utm_coords[0][1], utm_coords[1][3]
            )
            areas.append(utm_box.area)
        
        # Create GeoDataFrame with tile information
        gdf = gpd.GeoDataFrame({
            'geometry': geometries,
            'tile_id': range(len(tiles)),
            'area_m2': areas,
            'width_m': [((a/64.0)**0.5) for a in areas],  # Approximate width
            'height_m': [((a/64.0)**0.5) for a in areas]  # Approximate height
        }, crs="EPSG:4326")
        
        # Calculate statistics
        stats = {
            'min_area': min(areas),
            'max_area': max(areas),
            'mean_area': sum(areas) / len(areas),
            'total_tiles': len(tiles),
            'target_area': 64.0 * 64.0  # Target tile area (64m x 64m)
        }
        
        # Save files
        os.makedirs(output_dir, exist_ok=True)
        tiles_file = os.path.join(output_dir, f"{prefix}_preview.geojson")
        stats_file = os.path.join(output_dir, f"{prefix}_stats.txt")
        
        # Save GeoJSON
        gdf.to_file(tiles_file, driver='GeoJSON')
        
        # Save statistics
        with open(stats_file, 'w') as f:
            f.write("Tile Statistics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Tiles: {stats['total_tiles']}\n")
            f.write(f"Target Area: {stats['target_area']:.1f} m²\n")
            f.write(f"Min Area: {stats['min_area']:.1f} m²\n")
            f.write(f"Max Area: {stats['max_area']:.1f} m²\n")
            f.write(f"Mean Area: {stats['mean_area']:.1f} m²\n")
            f.write(f"Area Deviation: {(stats['mean_area'] - stats['target_area'])/stats['target_area']*100:.1f}%\n")
        
        print(f"\nTile preview saved to: {tiles_file}")
        print(f"Statistics saved to: {stats_file}")
        print("\nTile Statistics:")
        print("-" * 50)
        print(f"Total Tiles: {stats['total_tiles']}")
        print(f"Target Area: {stats['target_area']:.1f} m²")
        print(f"Min Area: {stats['min_area']:.1f} m²")
        print(f"Max Area: {stats['max_area']:.1f} m²")
        print(f"Mean Area: {stats['mean_area']:.1f} m²")
        print(f"Area Deviation: {(stats['mean_area'] - stats['target_area'])/stats['target_area']*100:.1f}%")
        
        return gdf