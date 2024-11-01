from owslib.wms import WebMapService
from PIL import Image
import io
import time
import concurrent.futures
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import os


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
        """Fetch single image with adaptive retry"""
        current_delay = retry_delay
        
        for attempt in range(max_retries):
            try:
                if self.wms is None and not self._connect():
                    time.sleep(current_delay)
                    current_delay *= 2  # Exponential backoff
                    continue

                img = self.wms.getmap(
                    layers=['Raster'],
                    srs='EPSG:4326',
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
                    current_delay *= 2  # Exponential backoff
                    self._connect()
                else:
                    print(f"\rFailed after {max_retries} attempts: {str(e)}\r", end='')
        return None

    def get_single_large_image(self, bbox, max_retries=3, retry_delay=1):
        """Fetch 280x280m image with adaptive retry"""
        current_delay = retry_delay
        
        for attempt in range(max_retries):
            try:
                if self.wms is None and not self._connect():
                    time.sleep(current_delay)
                    current_delay *= 2
                    continue

                img = self.wms.getmap(
                    layers=['Raster'],
                    srs='EPSG:4326',
                    bbox=bbox,
                    size=(2800, 2800),  # 280m at 10cm/px resolution
                    format='image/jpeg',
                    transparent=False,
                    timeout=self.timeout * 2  # Double timeout for larger images
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

    def fetch_batch(self, tile_bboxes, progress_bar=None):
        """Fetch batch with adaptive worker adjustment and backoff"""
        results = []
        failed_tiles = []
        self.current_workers = min(25, self.num_workers)  # Reduced to match chunk size
        max_batch_retries = 3
        backoff_time = 1

        for retry_attempt in range(max_batch_retries):
            futures = []
            current_tiles = failed_tiles if retry_attempt > 0 else tile_bboxes
            
            if retry_attempt > 0:
                print(f"\nRetrying {len(failed_tiles)} failed tiles with {self.current_workers} workers...")
                time.sleep(backoff_time)
                backoff_time *= 2
            
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.current_workers) as executor:
                    # Process exactly 25 tiles at once
                    chunk_size = 25
                    for i in range(0, len(current_tiles), chunk_size):
                        chunk = current_tiles[i:i + chunk_size]
                        
                        # 1 second break between chunks
                        if i > 0:
                            time.sleep(1.0)
                        
                        # Submit all tiles in this chunk at once
                        for bbox in chunk:
                            futures.append((executor.submit(self.get_single_image, bbox), bbox))

                        # Wait for this chunk to complete before moving to next
                        failed_tiles = []
                        chunk_futures = futures[-len(chunk):]
                        for future, bbox in chunk_futures:
                            try:
                                img = future.result(timeout=self.timeout)
                                if img is not None:
                                    results.append((img, bbox))
                                else:
                                    failed_tiles.append(bbox)
                            except concurrent.futures.TimeoutError:
                                failed_tiles.append(bbox)
                                self.current_workers = max(16, self.current_workers - 2)
                            except Exception as e:
                                failed_tiles.append(bbox)
                                print(f"\rConnection error for tile {bbox}: {str(e)}", end='\r', flush=True)
                            finally:
                                if progress_bar and retry_attempt == 0:
                                    progress_bar.update(1)

                    if not failed_tiles:
                        break

            except Exception as e:
                print(f"\rBatch error, retrying with fewer workers: {str(e)}")
                self.current_workers = max(16, self.current_workers - 8)
                time.sleep(backoff_time)
                if self.current_workers <= 16:
                    break

        if failed_tiles:
            # Log failed tiles to a file
            with open('failed_tiles.json', 'a') as f:
                json.dump({
                    'timestamp': time.time(),
                    'failed_tiles': [
                        {
                            'bbox': bbox,
                            'center': ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                        } 
                        for bbox in failed_tiles
                    ]
                }, f)
                f.write('\n')
            
            print(f"\nFailed tiles have been logged to 'failed_tiles.json'")
            print(f"Failed area centers (lon,lat):")
            for bbox in failed_tiles[:5]:  # Show first 5
                center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                print(f"  {center}")
            if len(failed_tiles) > 5:
                print(f"  ... and {len(failed_tiles)-5} more")
        
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

    def process_failed_tiles(self):
        """Process tiles that failed in previous runs"""
        if not os.path.exists('failed_tiles.json'):
            return []
        
        try:
            with open('failed_tiles.json', 'r') as f:
                failed_records = [json.loads(line) for line in f if line.strip()]
            
            # Collect all unique failed tiles
            all_failed_tiles = []
            for record in failed_records:
                all_failed_tiles.extend([tile['bbox'] for tile in record['failed_tiles']])
            all_failed_tiles = list(set(map(tuple, all_failed_tiles)))
            
            if all_failed_tiles:
                print(f"\nAttempting to process {len(all_failed_tiles)} previously failed tiles...")
                results = self.fetch_batch(all_failed_tiles)
                
                # Remove processed tiles from failed_tiles.json
                successful_tiles = set(bbox for img, bbox in results)
                remaining_failed = [
                    tile for tile in all_failed_tiles 
                    if tuple(tile) not in successful_tiles
                ]
                
                if remaining_failed:
                    with open('failed_tiles.json', 'w') as f:
                        json.dump({
                            'timestamp': time.time(),
                            'failed_tiles': [{'bbox': bbox} for bbox in remaining_failed]
                        }, f)
                else:
                    os.remove('failed_tiles.json')
                    
                return results
                
        except Exception as e:
            print(f"Error processing failed tiles: {str(e)}")
            return []