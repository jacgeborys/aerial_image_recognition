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
    def __init__(self, wms_url, layer='Raster', srs='EPSG:4326', size=(640, 640), 
                 image_format='image/png', timeout=45, num_workers=50):
        print("Initializing WMS connection...")
        self.wms_url = wms_url
        self.timeout = timeout
        self.num_workers = num_workers
        self.wms = None
        self.session = self._create_session()
        
        # WMS configuration
        self.layer = layer
        self.srs = srs
        self.size = size
        self.image_format = image_format
        
        self.stats = {
            'attempts': 0,
            'timeouts': 0,
            'timeout_attempts': 0,
            'successes': 0,
            'failures': 0,
            'total_bytes': 0,
            'start_time': time.time(),
            'connection_times': [],
            'retry_counts': [],
            'status_codes': []
        }
        
        if not self._connect():
            raise RuntimeError("Failed to establish WMS connection")
        print("WMS connection established successfully")

    def _create_session(self):
        """Create session with conservative settings"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
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

    def get_single_image(self, bbox, max_retries=3, initial_delay=0.1):
        """Fetch single image with proper WMS response handling"""
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
                    # Try to open the image directly from the response
                    img = Image.open(BytesIO(response.read()))
                    self.stats['successes'] += 1
                    return img
                except Exception as img_error:
                    print(f"\nError reading image from response: {str(img_error)}")
                    self.stats['failures'] += 1
                
                # Calculate backoff delay: 0.1s -> 0.2s -> 0.4s
                delay = initial_delay * (2 ** attempt)
                time.sleep(delay)
                
            except Exception as e:
                if 'timeout' in str(e).lower():
                    self.stats['timeout_attempts'] += 1
                self.stats['failures'] += 1
                print(f"\nError fetching tile {bbox}: {str(e)}")
                delay = initial_delay * (2 ** attempt)
                time.sleep(delay)
                continue
        
        self.stats['timeouts'] += 1
        return None

    def fetch_batch(self, tiles, progress_bar=None):
        """Parallel fetch with optimized delays and retries"""
        if not tiles:
            return []
        
        print("\nStarting batch download...")
        self._print_stats()
        
        results = []
        failed_tiles = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            # Submit initial batch
            for bbox in tiles:
                futures.append((executor.submit(self.get_single_image, bbox), bbox))
            
            # Process results
            for future, bbox in futures:
                try:
                    time.sleep(0.03)  # Reduced sleep time
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
                    print(f"\nTimeout for tile {bbox}: {str(e)}")
                    self._print_stats()
        
        # Retry failed tiles with increasing delays
        if failed_tiles:
            print(f"\nRetrying {len(failed_tiles)} failed tiles...")
            retry_delays = [0.5, 1, 2]  # Progressive delays for retries
            
            for bbox in failed_tiles:
                for delay in retry_delays:
                    time.sleep(delay)
                    img = self.get_single_image(bbox, max_retries=1)  # Single retry with longer delay
                    if img is not None:
                        results.append((img, bbox))
                        if progress_bar:
                            progress_bar.update(0)
                        break  # Success, move to next tile
        
        print("\nBatch complete. Final stats:")
        self._print_stats()
        print()
        
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