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


class WMSHandler:
    def __init__(self, wms_url, num_workers=32, timeout=30):
        print("Initializing WMS connection...")
        self.wms_url = wms_url
        self.num_workers = num_workers
        self.timeout = timeout
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

                img = self.wms.getmap(
                    layers=['Raster'],
                    srs='EPSG:4326',  # Use WGS84 for WMS request
                    bbox=bbox,
                    size=(640, 640),
                    format='image/jpeg',
                    transparent=False,
                    timeout=self.timeout
                )
                return Image.open(io.BytesIO(img.read())).convert('RGB')

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