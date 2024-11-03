from owslib.wms import WebMapService
from PIL import Image
import io
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


class WMSHandler:
    def __init__(self, wms_url, num_workers=32, timeout=30, resolution=0.1):
        print("Initializing WMS connection...")
        self.wms_url = wms_url
        self.num_workers = num_workers
        self.timeout = timeout
        self.resolution = resolution  # meters per pixel
        self.wms = None
        self.session = self._create_session()
        self.min_workers = 8  # New: minimum workers
        self.current_workers = num_workers  # New: track current workers
        if not self._connect():
            raise RuntimeError("Failed to establish WMS connection")
        print("WMS connection established successfully")

    def _create_session(self):
        """Create session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.num_workers,
            pool_maxsize=self.num_workers
        )
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

    def get_single_image(self, bbox, max_retries=3, retry_delay=1):
        """Fetch single image with proper WGS84 coordinates"""
        current_delay = retry_delay
        
        for attempt in range(max_retries):
            try:
                if self.wms is None and not self._connect():
                    time.sleep(current_delay)
                    current_delay *= 2
                    continue

                # Calculate required size for target resolution
                utm_zone = int((bbox[0] + 180) / 6) + 1
                transformer = Transformer.from_crs("EPSG:4326", f"EPSG:326{utm_zone}", always_xy=True)
                x1, y1 = transformer.transform(bbox[0], bbox[1])
                x2, y2 = transformer.transform(bbox[2], bbox[3])
                width_meters = abs(x2 - x1)
                height_meters = abs(y2 - y1)
                
                # Request image based on resolution
                target_width = int(width_meters / self.resolution)
                target_height = int(height_meters / self.resolution)
                
                # Cap at reasonable size and maintain aspect ratio
                min_size = 3200
                max_size = 3000
                if target_width > max_size or target_height > max_size:
                    scale = max_size / max(target_width, target_height)
                    target_width = int(target_width * scale)
                    target_height = int(target_height * scale)
                elif target_width < min_size or target_height < min_size:
                    scale = min_size / min(target_width, target_height)
                    target_width = int(target_width * scale)
                    target_height = int(target_height * scale)

                img = self.wms.getmap(
                    layers=['Raster'],
                    srs='EPSG:4326',
                    bbox=bbox,
                    size=(target_width, target_height),
                    format='image/jpeg',
                    transparent=False,
                    timeout=self.timeout
                )
                
                # Resize to 640x640 for model input
                image = Image.open(io.BytesIO(img.read())).convert('RGB')
                return image.resize((640, 640), Image.Resampling.LANCZOS)

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(current_delay)
                    current_delay *= 2
                    self._connect()
                else:
                    print(f"\rFailed after {max_retries} attempts: {str(e)}\r", end='')
        return None

    def fetch_batch(self, tiles, progress_bar=None):
        """Fetch a batch of tiles in parallel"""
        if not tiles:
            return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            results = []
            
            # Submit all tiles for processing
            for bbox in tiles:
                futures.append((executor.submit(self.get_single_image, bbox), bbox))
            
            # Process results as they complete
            for future, bbox in futures:
                try:
                    img = future.result(timeout=self.timeout)
                    if img is not None:
                        results.append((img, bbox))
                    if progress_bar:
                        progress_bar.update(1)
                except concurrent.futures.TimeoutError:
                    print(f"\rTimeout fetching tile {bbox}", end='', flush=True)
                except Exception as e:
                    print(f"\rError fetching tile {bbox}: {str(e)}", end='', flush=True)

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