from owslib.wmts import WebMapTileService
import geopandas as gpd
from shapely.geometry import box
import os
from datetime import datetime
import requests
from PIL import Image
import io
from pyproj import Transformer
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import math
import concurrent.futures

class WMTSHandler:
    def __init__(self, num_workers=32, timeout=30):
        self.wmts_url = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMTS/StandardResolution"
        self.num_workers = num_workers
        self.timeout = timeout
        
        # Add session and stats tracking
        self.session = self._create_session()
        self.stats = {
            'attempts': 0,
            'retries': 0,
            'successes': 0,
            'total_failures': 0,
            'retry_time': 0
        }
        
        # Initialize WMTS
        self.wmts = WebMapTileService(
            url=f"{self.wmts_url}?service=WMTS&request=GetCapabilities&version=1.0.0"
        )
        
        # Set service parameters
        self.layer = 'ORTOFOTOMAPA'
        self.matrix_set = 'EPSG:2180'
        self.zoom_level = 'EPSG:2180:15'
        self.format = 'image/jpeg'
        
        # Get and print matrix details
        matrix = self.wmts.tilematrixsets[self.matrix_set].tilematrix[self.zoom_level]
        print(f"\nMatrix Details for {self.zoom_level}:")
        print(f"Matrix Width: {matrix.matrixwidth}")
        print(f"Matrix Height: {matrix.matrixheight}")
        print(f"Resolution: ~10.6cm/pixel (after resize to 640x640)")
        print(f"Coverage per tile: {67.7:.1f}m x {67.7:.1f}m")
        
        # Create directory for GeoJSON files
        self.geojson_dir = "debug_geojson"
        os.makedirs(self.geojson_dir, exist_ok=True)

        self.test_high_zoom_tile()  # Test tile fetching

    def _create_session(self):
        """Create session with optimized connection pooling"""
        session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.num_workers,
            pool_maxsize=self.num_workers * 2,
            pool_block=False
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def save_batch_footprint(self, tiles, batch_idx):
        """Save batch footprint as GeoJSON"""
        try:
            # Create geometries
            geometries = []
            for bbox in tiles:
                minx, miny, maxx, maxy = bbox
                geometries.append(box(minx, miny, maxx, maxy))
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame({
                'geometry': geometries,
                'tile_id': range(len(tiles)),
                'minx': [t[0] for t in tiles],
                'miny': [t[1] for t in tiles],
                'maxx': [t[2] for t in tiles],
                'maxy': [t[3] for t in tiles]
            }, crs="EPSG:4326")
            
            # Save GeoJSON
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            geojson_path = os.path.join(self.geojson_dir, f"tiles_{timestamp}.geojson")
            gdf.to_file(geojson_path, driver='GeoJSON')
            print(f"Saved tile preview: {geojson_path}")
            
            return geojson_path
            
        except Exception as e:
            print(f"Error saving tile preview: {str(e)}")
            return None

    def get_tile_for_bbox(self, bbox, max_retries=2, retry_delay=1.5):
        """Get tile covering the bbox with retry logic"""
        try:
            # bbox comes in WGS84 (EPSG:4326) coordinates
            center_lon = (bbox[0] + bbox[2]) / 2
            center_lat = (bbox[1] + bbox[3]) / 2
            
            # Transform to EPSG:2180
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
            x, y = transformer.transform(center_lon, center_lat)
            
            # Get matrix parameters
            matrix = self.wmts.tilematrixsets[self.matrix_set].tilematrix[self.zoom_level]
            top_left_x, top_left_y = matrix.topleftcorner
            
            # Calculate tile size
            tile_width = (matrix.scaledenominator * 0.00028) * matrix.tilewidth
            tile_height = (matrix.scaledenominator * 0.00028) * matrix.tileheight
            
            # Calculate tile indices
            col = int((x - top_left_x) / tile_width)
            row = int((top_left_y - y) / tile_height)
            
            # Detailed debugging for first few tiles
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
            
            if self._debug_count < 5:
                print(f"\nTile Debug [{self._debug_count}]:")
                print(f"Input bbox: {bbox}")
                print(f"Center WGS84: {center_lon:.6f}, {center_lat:.6f}")
                print(f"EPSG:2180: {x:.1f}, {y:.1f}")
                print(f"Matrix top-left: {top_left_x:.1f}, {top_left_y:.1f}")
                print(f"Tile size: {tile_width:.1f}m x {tile_height:.1f}m")
                print(f"Calculated indices: col={col}, row={row}")
                print(f"Matrix bounds: {matrix.matrixwidth}x{matrix.matrixheight}")
                self._debug_count += 1
            
            # Bounds checking
            if col < 0 or row < 0 or col >= matrix.matrixwidth or row >= matrix.matrixheight:
                if self._debug_count < 5:
                    print(f"Out of bounds: col={col}, row={row}")
                return None, None
            
            # Construct URL and fetch tile
            url = (
                f"{self.wmts_url}"
                f"?service=WMTS&request=GetTile&version=1.0.0"
                f"&layer={self.layer}&style=default&format={self.format}"
                f"&tileMatrixSet={self.matrix_set}&tileMatrix={self.zoom_level}"
                f"&tileRow={row}&tileCol={col}"
            )
            
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                img = img.resize((640, 640), Image.Resampling.LANCZOS)
                
                # Save first few tiles for verification
                if self._debug_count <= 5:
                    debug_dir = "test_tiles/debug"
                    os.makedirs(debug_dir, exist_ok=True)
                    img.save(f"{debug_dir}/tile_{self._debug_count-1}.jpg")
                    
                    # Save tile metadata
                    with open(f"{debug_dir}/tile_{self._debug_count-1}_meta.txt", "w") as f:
                        f.write(f"URL: {url}\n")
                        f.write(f"Tile indices: col={col}, row={row}\n")
                        f.write(f"EPSG:2180: {x:.1f}, {y:.1f}\n")
                        f.write(f"WGS84: {center_lon:.6f}, {center_lat:.6f}\n")
                
                return img, bbox
                
            return None, None
            
        except Exception as e:
            print(f"Error in get_tile_for_bbox: {str(e)}")
            return None, None

    def _print_stats(self):
        """Print connection statistics"""
        if self.stats['attempts'] > 0:
            success_rate = (self.stats['successes'] / self.stats['attempts']) * 100
            retry_rate = (self.stats['retries'] / self.stats['attempts']) * 100
            avg_retry_time = self.stats['retry_time'] / (self.stats['retries'] or 1)
            
            print(f"\rWMTS: {self.stats['successes']}/{self.stats['attempts']} OK ({success_rate:.1f}%) | "
                  f"Retries: {self.stats['retries']} ({retry_rate:.1f}%) | "
                  f"Avg retry: {avg_retry_time:.1f}s", end='')

    def _geo_to_tile(self, lon, lat):
        """Convert geographic coordinates to tile coordinates using WMTS standard"""
        try:
            matrix = self.wmts.tilematrixsets[self.matrix_set].tilematrix[self.zoom_level]
            
            print(f"\nCoordinate conversion for lon={lon}, lat={lat}")
            print(f"Matrix top left: {matrix.topleftcorner}")
            print(f"Matrix size: {matrix.matrixwidth}x{matrix.matrixheight}")
            
            # WMTS uses normalized coordinates relative to [-180,180] x [-90,90]
            # First normalize coordinates to [0,1]
            norm_x = (lon - (-180.0)) / 360.0
            norm_y = (90.0 - lat) / 180.0
            
            # Calculate tile coordinates
            col = int(norm_x * matrix.matrixwidth)
            row = int(norm_y * matrix.matrixheight)
            
            print(f"Normalized coordinates: x={norm_x:.6f}, y={norm_y:.6f}")
            print(f"Calculated tile coordinates: col={col}, row={row}")
            
            # Ensure within bounds
            if col < 0 or col >= matrix.matrixwidth or row < 0 or row >= matrix.matrixheight:
                print("Warning: Coordinates outside matrix bounds")
                return None, None
                
            return col, row
            
        except Exception as e:
            print(f"Error in _geo_to_tile: {str(e)}")
            raise

    def _get_tile_bbox(self, col, row):
        """Calculate the actual bbox for a tile"""
        try:
            # Get matrix parameters
            matrix = self.wmts.tilematrixsets[self.matrix_set].tilematrix[self.zoom_level]
            
            # Calculate tile bounds in degrees
            lon_per_tile = 360.0 / matrix.matrixwidth
            lat_per_tile = 180.0 / matrix.matrixheight
            
            min_lon = -180.0 + (col * lon_per_tile)
            max_lon = min_lon + lon_per_tile
            max_lat = 90.0 - (row * lat_per_tile)
            min_lat = max_lat - lat_per_tile
            
            return (min_lon, min_lat, max_lon, max_lat)
            
        except Exception as e:
            print(f"Error in _get_tile_bbox: {str(e)}")
            raise

    def test_tile_range(self):
        """Test a range of tile coordinates to find valid ones"""
        # Test a range around Warsaw's approximate location
        test_coords = [
            # Center coordinates for testing
            (20.0, 52.0),  # Base coordinates
            
            # Test different zoom levels
            {'col': 5000, 'row': 1000},
            {'col': 10000, 'row': 2000},
            {'col': 15000, 'row': 3000},
            {'col': 20000, 'row': 4000},
            {'col': 25000, 'row': 5000}
        ]
        
        print("\nTesting tile coordinate ranges...")
        
        # First test with coordinates
        lon, lat = test_coords[0]
        print(f"\nTesting geographic coordinates: {lon}, {lat}")
        col, row = self._geo_to_tile(lon, lat)
        if col is not None and row is not None:
            print(f"Converted to tile coordinates: col={col}, row={row}")
            self._test_single_tile(col, row)
        
        # Then test specific tile coordinates
        for coords in test_coords[1:]:
            col, row = coords['col'], coords['row']
            print(f"\nTesting tile coordinates: col={col}, row={row}")
            self._test_single_tile(col, row)

    def _test_single_tile(self, col, row):
        """Test fetching a single tile with given coordinates"""
        tile_url = (
            f"{self.wmts_url}"
            f"?service=WMTS"
            f"&request=GetTile"
            f"&version=1.0.0"
            f"&layer={self.layer}"
            f"&style=default"
            f"&format={self.format}"
            f"&tileMatrixSet={self.matrix_set}"
            f"&tileMatrix={self.zoom_level}"
            f"&tileRow={row}"
            f"&tileCol={col}"
        )
        
        print(f"Requesting URL: {tile_url}")
        
        try:
            response = requests.get(tile_url, timeout=self.timeout)
            print(f"Status: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type')}")
            print(f"Content-Length: {len(response.content)}")
            
            if 'text/xml' in response.headers.get('content-type', ''):
                print("Error response:")
                print(response.text)
            elif len(response.content) > 0:
                with open(f"test_tile_{col}_{row}.png", "wb") as f:
                    f.write(response.content)
                print(f"Saved test tile to: test_tile_{col}_{row}.png")
        except Exception as e:
            print(f"Error testing tile: {str(e)}")

    def test_high_zoom_tile(self):
        """Test fetching tiles at different zoom levels"""
        # Test zoom levels from 13 to 16
        zoom_levels = [f'EPSG:2180:{i}' for i in range(13, 17)]
        
        for zoom_level in zoom_levels:
            self.zoom_level = zoom_level
            matrix = self.wmts.tilematrixsets[self.matrix_set].tilematrix[self.zoom_level]
            
            # Calculate center tile coordinates (using the successful approach from test_wmts.py)
            col = matrix.matrixwidth // 2
            row = matrix.matrixheight // 2
            
            # Get matrix parameters
            top_left_x, top_left_y = matrix.topleftcorner
            
            # Calculate the actual coordinates for this tile
            tile_width = (matrix.scaledenominator * 0.00028) * matrix.tilewidth
            tile_height = (matrix.scaledenominator * 0.00028) * matrix.tileheight
            
            print(f"\n=== Testing zoom level {zoom_level} ===")
            print(f"Matrix dimensions: {matrix.matrixwidth}x{matrix.matrixheight}")
            print(f"Center tile: col={col}, row={row}")
            print(f"Tile size: {tile_width:.1f}m x {tile_height:.1f}m")
            
            # Create directory for test tiles if it doesn't exist
            os.makedirs("test_tiles", exist_ok=True)
            
            response = self.session.get(
                f"{self.wmts_url}"
                f"?service=WMTS&request=GetTile&version=1.0.0"
                f"&layer={self.layer}&style=default&format={self.format}"
                f"&tileMatrixSet={self.matrix_set}"
                f"&tileMatrix={self.zoom_level}"
                f"&tileRow={row}&tileCol={col}",
                timeout=self.timeout
            )
            print(f"Calculated tile: col={col}, row={row}")
            if response.status_code == 200:
                # Save tile with zoom level in filename
                zoom_number = zoom_level.split(':')[-1]
                filename = f"test_tiles/warsaw_z{zoom_number.zfill(2)}.jpg"
                
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"✓ Saved tile to: {filename}")
                print(f"  File size: {len(response.content)/1024:.1f}KB")
                
                # Also save resized version for debugging
                img = Image.open(io.BytesIO(response.content))
                resized = img.resize((640, 640), Image.Resampling.LANCZOS)
                resized.save(f"test_tiles/warsaw_z{zoom_number.zfill(2)}_resized.jpg")
            else:
                print(f"✗ Error: {response.status_code}")
                print(response.text)

    def fetch_batch(self, tiles, progress_bar=None):
        """Fetch a batch of tiles with adaptive timeouts"""
        if not tiles:
            return []

        request_times = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            results = []
            
            # Submit initial batch
            for bbox in tiles[:self.num_workers]:
                futures.append((executor.submit(self.get_tile_for_bbox, bbox), bbox))
            
            # Process results and submit new tasks adaptively
            tiles_iter = iter(tiles[self.num_workers:])
            for future, bbox in futures:
                try:
                    start_time = time.time()
                    result = future.result(timeout=self.timeout)
                    if result is not None:
                        img, bbox = result
                        results.append((img, bbox))
                        request_times.append(time.time() - start_time)
                        
                        # Submit next tile if available
                        try:
                            next_bbox = next(tiles_iter)
                            futures.append((executor.submit(self.get_tile_for_bbox, next_bbox), next_bbox))
                        except StopIteration:
                            pass
                        
                        if progress_bar:
                            progress_bar.update(1)
                        
                except concurrent.futures.TimeoutError:
                    print(f"\rTimeout fetching tile {bbox}", end='', flush=True)
                except Exception as e:
                    print(f"\rError fetching tile {bbox}: {str(e)}", end='', flush=True)

        return results

    def test_warsaw_area(self):
        """Test fetching tiles covering Warsaw central area with detailed debugging"""
        # Read the shapefile
        warsaw = gpd.read_file("warsaw_central.shp")
        print("\nInput shapefile CRS:", warsaw.crs)
        
        # Get centroid of the shape in original CRS
        centroid = warsaw.geometry.unary_union.centroid
        print(f"Shapefile centroid (original CRS): ({centroid.x:.6f}, {centroid.y:.6f})")
        
        # Transform to EPSG:2180
        if warsaw.crs != "EPSG:2180":
            warsaw = warsaw.to_crs("EPSG:2180")
            
        # Get bounds and centroid in EPSG:2180
        minx, miny, maxx, maxy = warsaw.total_bounds
        centroid_2180 = warsaw.geometry.unary_union.centroid
        
        print("\nCoordinate Information:")
        print("-" * 50)
        print("EPSG:2180 bounds:")
        print(f"X: {minx:.1f} - {maxx:.1f}")
        print(f"Y: {miny:.1f} - {maxy:.1f}")
        print(f"Centroid: ({centroid_2180.x:.1f}, {centroid_2180.y:.1f})")
        
        # Create transformers for verification
        transformer_to_2180 = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
        transformer_to_4326 = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
        
        # Verify transformations
        wgs_x, wgs_y = transformer_to_4326.transform(centroid_2180.x, centroid_2180.y)
        print(f"\nCentroid in WGS84: ({wgs_x:.6f}, {wgs_y:.6f})")
        
        # Test zoom levels
        zoom_levels = [f'EPSG:2180:{i}' for i in range(13, 17)]
        
        for zoom_level in zoom_levels:
            self.zoom_level = zoom_level
            matrix = self.wmts.tilematrixsets[self.matrix_set].tilematrix[self.zoom_level]
            
            # Get matrix parameters
            top_left_x, top_left_y = matrix.topleftcorner
            tile_width = (matrix.scaledenominator * 0.00028) * matrix.tilewidth
            tile_height = (matrix.scaledenominator * 0.00028) * matrix.tileheight
            
            # Calculate tile for centroid
            col = int((centroid_2180.x - top_left_x) / tile_width)
            row = int((top_left_y - centroid_2180.y) / tile_height)
            
            print(f"\n=== Testing zoom level {zoom_level} ===")
            print(f"Matrix bounds: {matrix.matrixwidth}x{matrix.matrixheight}")
            print(f"Top left corner: ({top_left_x:.1f}, {top_left_y:.1f})")
            print(f"Tile size: {tile_width:.1f}m x {tile_height:.1f}m")
            print(f"Centroid tile: col={col}, row={row}")
            
            # Calculate actual coordinates of the tile for verification
            tile_x = top_left_x + (col * tile_width)
            tile_y = top_left_y - (row * tile_height)
            tile_wgs_x, tile_wgs_y = transformer_to_4326.transform(tile_x, tile_y)
            
            print(f"Tile center coordinates:")
            print(f"EPSG:2180: ({tile_x:.1f}, {tile_y:.1f})")
            print(f"WGS84: ({tile_wgs_x:.6f}, {tile_wgs_y:.6f})")
            
            # Create directory for test tiles
            os.makedirs("test_tiles/warsaw_debug", exist_ok=True)
            
            # Fetch the centroid tile
            response = self.session.get(
                f"{self.wmts_url}"
                f"?service=WMTS&request=GetTile&version=1.0.0"
                f"&layer={self.layer}&style=default&format={self.format}"
                f"&tileMatrixSet={self.matrix_set}&tileMatrix={self.zoom_level}"
                f"&tileRow={row}&tileCol={col}",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                # Save both original and debug info
                zoom_number = zoom_level.split(':')[-1]
                filename = f"test_tiles/warsaw_debug/centroid_z{zoom_number.zfill(2)}.jpg"
                debug_filename = f"test_tiles/warsaw_debug/centroid_z{zoom_number.zfill(2)}_debug.txt"
                
                with open(filename, "wb") as f:
                    f.write(response.content)
                
                # Save debug info
                with open(debug_filename, "w") as f:
                    f.write(f"Zoom Level: {zoom_level}\n")
                    f.write(f"Tile Position: col={col}, row={row}\n")
                    f.write(f"EPSG:2180 Coordinates: ({tile_x:.1f}, {tile_y:.1f})\n")
                    f.write(f"WGS84 Coordinates: ({tile_wgs_x:.6f}, {tile_wgs_y:.6f})\n")
                    f.write(f"File Size: {len(response.content)/1024:.1f}KB\n")
                
                print(f"✓ Saved tile and debug info to: {filename}")
            else:
                print(f"✗ Error: {response.status_code}")
                print(response.text)

    def test_warsaw_coordinates(self):
        """Test tile fetching specifically for Warsaw center"""
        # Warsaw center coordinates (WGS84)
        warsaw_lon, warsaw_lat = 21.0122, 52.2297
        
        # Transform to EPSG:2180
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
        x, y = transformer.transform(warsaw_lon, warsaw_lat)
        
        print(f"\nWarsaw Center:")
        print(f"WGS84: {warsaw_lon:.4f}°E, {warsaw_lat:.4f}°N")
        print(f"EPSG:2180: {x:.1f}, {y:.1f}")
        
        # Test zoom levels
        for zoom_level in [f'EPSG:2180:{i}' for i in range(13, 17)]:
            self.zoom_level = zoom_level
            matrix = self.wmts.tilematrixsets[self.matrix_set].tilematrix[self.zoom_level]
            
            # Calculate tile coordinates
            top_left_x, top_left_y = matrix.topleftcorner
            tile_width = (matrix.scaledenominator * 0.00028) * matrix.tilewidth
            tile_height = (matrix.scaledenominator * 0.00028) * matrix.tileheight
            
            col = int((x - top_left_x) / tile_width)
            row = int((top_left_y - y) / tile_height)
            
            print(f"\n=== Zoom {zoom_level} ===")
            print(f"Matrix size: {matrix.matrixwidth}x{matrix.matrixheight}")
            print(f"Warsaw tile: col={col}, row={row}")
            
            # Save test tile
            os.makedirs("test_tiles/warsaw_center", exist_ok=True)
            response = self.session.get(
                f"{self.wmts_url}?service=WMTS&request=GetTile&version=1.0.0"
                f"&layer={self.layer}&style=default&format={self.format}"
                f"&tileMatrixSet={self.matrix_set}&tileMatrix={self.zoom_level}"
                f"&tileRow={row}&tileCol={col}",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                filename = f"test_tiles/warsaw_center/z{zoom_level.split(':')[-1]}.jpg"
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"✓ Saved tile: {filename}")
            else:
                print(f"✗ Error: {response.status_code}")
                print(response.text)

    def verify_matrix_parameters(self):
        """Verify WMTS matrix parameters for Warsaw area"""
        # Test coordinates for Warsaw center (Palace of Culture)
        warsaw_lon, warsaw_lat = 21.0122, 52.2297
        
        # Transform to EPSG:2180
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
        x, y = transformer.transform(warsaw_lon, warsaw_lat)
        
        print("\nWarsaw Center Point:")
        print(f"WGS84: {warsaw_lon:.6f}°E, {warsaw_lat:.6f}°N")
        print(f"EPSG:2180: {x:.1f}, {y:.1f}")
        
        # Print matrix parameters for each zoom level
        for zoom_level in [f'EPSG:2180:{i}' for i in range(13, 17)]:
            matrix = self.wmts.tilematrixsets[self.matrix_set].tilematrix[zoom_level]
            print(f"\nZoom Level {zoom_level}:")
            print(f"Top Left Corner: {matrix.topleftcorner}")
            print(f"Matrix Size: {matrix.matrixwidth}x{matrix.matrixheight}")
            print(f"Tile Size: {matrix.tilewidth}x{matrix.tileheight}")
            print(f"Scale Denominator: {matrix.scaledenominator}")

    def test_single_tile(self):
        """Test fetching a single tile for Warsaw center"""
        # Warsaw center coordinates (known good coordinates)
        warsaw_lon, warsaw_lat = 20.966395, 52.266797  # From your debug output
        
        # Transform to EPSG:2180
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
        x, y = transformer.transform(warsaw_lon, warsaw_lat)
        
        # Get matrix parameters for zoom level 15
        matrix = self.wmts.tilematrixsets[self.matrix_set].tilematrix['EPSG:2180:15']
        
        # IMPORTANT: The matrix origin is at the top-left corner of Poland's EPSG:2180 zone
        # These are the correct parameters for EPSG:2180 in Poland
        matrix_origin_x = 420000.0  # Approximate western edge of Poland in EPSG:2180
        matrix_origin_y = 850000.0  # Approximate northern edge of Poland in EPSG:2180
        
        tile_width = (matrix.scaledenominator * 0.00028) * matrix.tilewidth
        tile_height = (matrix.scaledenominator * 0.00028) * matrix.tileheight
        
        # Calculate tile indices relative to the correct origin
        col = int((x - matrix_origin_x) / tile_width)
        row = int((matrix_origin_y - y) / tile_height)
        
        print(f"\nCoordinate Debug:")
        print(f"WGS84: {warsaw_lon:.6f}°E, {warsaw_lat:.6f}°N")
        print(f"EPSG:2180: {x:.1f}, {y:.1f}")
        print(f"Matrix origin: {matrix_origin_x:.1f}, {matrix_origin_y:.1f}")
        print(f"Tile size: {tile_width:.1f}m x {tile_height:.1f}m")
        print(f"Calculated indices: col={col}, row={row}")
        print(f"Matrix bounds: {matrix.matrixwidth}x{matrix.matrixheight}")
        
        # Verify the indices are within bounds
        if 0 <= col < matrix.matrixwidth and 0 <= row < matrix.matrixheight:
            response = self.session.get(
                f"{self.wmts_url}?service=WMTS&request=GetTile&version=1.0.0"
                f"&layer={self.layer}&style=default&format={self.format}"
                f"&tileMatrixSet={self.matrix_set}&tileMatrix=EPSG:2180:15"
                f"&tileRow={row}&tileCol={col}"
            )
            
            if response.status_code == 200:
                filename = f"test_tiles/debug/warsaw_test.jpg"
                os.makedirs("test_tiles/debug", exist_ok=True)
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"\n✓ Saved tile to: {filename}")
                
                # Calculate actual coordinates of this tile's center
                tile_center_x = matrix_origin_x + (col * tile_width) + (tile_width / 2)
                tile_center_y = matrix_origin_y - (row * tile_height) - (tile_height / 2)
                center_lon, center_lat = transformer.transform(tile_center_x, tile_center_y, direction='INVERSE')
                print(f"Tile center coordinates:")
                print(f"EPSG:2180: {tile_center_x:.1f}, {tile_center_y:.1f}")
                print(f"WGS84: {center_lon:.6f}°E, {center_lat:.6f}°N")
        else:
            print(f"\n✗ Calculated tile indices out of bounds")

    def debug_coordinates(self):
        """Debug coordinate systems and transformations"""
        print("\n=== Coordinate Systems Debug ===")
        
        # Read shapefile
        shapefile_path = r"C:\Users\Asus\OneDrive\Pulpit\Rozne\QGIS\car_recognition\gis\frames\warsaw_central.shp"
        warsaw = gpd.read_file(shapefile_path)
        
        print(f"\n1. Shapefile Information:")
        print(f"CRS: {warsaw.crs}")
        print(f"Bounds: {warsaw.total_bounds}")
        centroid = warsaw.geometry.union_all().centroid
        print(f"Centroid: {centroid.x:.1f}, {centroid.y:.1f}")
        
        # WMTS Matrix Information
        print("\n2. WMTS Matrix Information:")
        wmts = WebMapTileService("https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMTS/StandardResolution")
        matrix_set = wmts.tilematrixsets['EPSG:3857']
        
        for zoom in [13, 14, 15, 16]:
            zoom_id = f'EPSG:3857:{zoom}'
            matrix = matrix_set.tilematrix[zoom_id]
            resolution = matrix.scaledenominator * 0.00028
            print(f"\nZoom Level {zoom_id}:")
            print(f"Matrix size: {matrix.matrixwidth}x{matrix.matrixheight}")
            print(f"Ground resolution: {resolution:.2f}m/px")
            print(f"Pixel size at ground: {resolution:.2f}m")