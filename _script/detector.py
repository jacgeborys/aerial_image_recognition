import os
import time
from tqdm import tqdm
import geopandas as gpd
import gc
import traceback
import torch
from PIL import ImageEnhance
from datetime import datetime

from .wms_handler import WMSHandler
from .gpu_handler import GPUHandler
from .utils import TileGenerator, CheckpointManager, ResultsManager
from .monitors import GPUMonitor
from .config import DEFAULT_CONFIG


class CarDetector:
    def __init__(self, base_dir, custom_config=None):
        """Initialize detector with configuration"""
        try:
            print("\nInitializing detector...")
            self.base_dir = base_dir
            self.config = self._load_config(custom_config)
            
            # Set up paths using base_dir
            self._setup_paths()
            
            # Initialize components with unified config
            self._initialize_components()
            
        except Exception as e:
            print(f"Error initializing detector: {str(e)}")
            raise

    def _load_config(self, custom_config=None):
        """Load configuration with defaults and custom overrides"""
        config = DEFAULT_CONFIG.copy()
        if custom_config:
            config.update(custom_config)
        return config

    def _setup_paths(self):
        """Set up all paths relative to base_dir"""
        frame_name = os.path.splitext(self.config['frame_path'])[0]
        self.frame_path = os.path.join(self.base_dir, 'gis', 'frames', self.config['frame_path'])
        self.output_dir = os.path.join(self.base_dir, 'output', frame_name)
        self.model_path = os.path.join(self.base_dir, 'models', self.config['model_path'])
        os.makedirs(self.output_dir, exist_ok=True)

    def _initialize_components(self):
        """Initialize all components with unified configuration"""
        print("\nInitializing WMS connection and GPU...")
        self.wms_handler = WMSHandler(
            wms_url=self.config['wms_url'],
            layer=self.config['wms_layer'],
            srs=self.config['wms_srs'],
            size=self.config['wms_size'],
            image_format=self.config['wms_format'],
            timeout=self.config.get('timeout', 45),
            num_workers=self.config['num_workers']
        )
        
        self.gpu_handler = GPUHandler(
            model_path=self.model_path,
            confidence_threshold=self.config['confidence_threshold'],
            max_gpu_memory=self.config['max_gpu_memory']
        )
        
        self.checkpoint_manager = CheckpointManager(self.output_dir)
        self.results_manager = ResultsManager(
            self.output_dir,
            prefix=self.config['output_prefix'],
            duplicate_distance=self.config['duplicate_distance']
        )

    def process_images(self, image_batch, progress_bar=None):
        """Process a batch of images through GPU"""
        try:
            # Ensure GPU handler is initialized
            if not self.gpu_handler:
                self._initialize_handlers()

            # Process batch
            detections = self.gpu_handler.process_batch(
                image_batch,
                queue_size=self.config['queue_size']
            )

            return detections
        except Exception as e:
            print(f"\nError processing batch: {str(e)}")
            return []

    def fetch_images(self, tile_batch, progress_bar=None):
        """Fetch a batch of images from WMS"""
        try:
            # Ensure WMS handler is initialized
            if not self.wms_handler:
                self._initialize_handlers()

            # Fetch images
            return self.wms_handler.fetch_batch(tile_batch, progress_bar)
        except Exception as e:
            print(f"\nError fetching images: {str(e)}")
            return []

    def _process_batch(self, batch_tiles, processed_count, total_tiles):
        """Process a single batch with clear status reporting"""
        batch_size = len(batch_tiles)
        batch_start = time.time()

        # Print minimal batch header
        current_batch = (processed_count // batch_size) + 1
        total_batches = (total_tiles + batch_size - 1) // batch_size
        print(f"\nBatch {current_batch}/{total_batches}")
        print("-" * 50)
        
        with tqdm(
            total=batch_size,
            desc="Downloading",
            leave=False,
            bar_format='{desc:<12} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ) as fetch_progress:
            images = self.wms_handler.fetch_batch(batch_tiles, fetch_progress)

        fetch_time = time.time() - batch_start
        
        # Compact fetch summary
        print(f"Downloaded: {len(images)}/{batch_size} ({len(images)/batch_size*100:.1f}%) in {fetch_time:.1f}s")

        if images:
            yolo_start = time.time()
            batch_detections = self.gpu_handler.process_batch(
                images,
                queue_size=self.config.get('queue_size', self.config['batch_size'])
            )
            yolo_time = time.time() - yolo_start
            
            # Compact YOLO summary
            print(f"Detections: {len(batch_detections)} in {yolo_time:.1f}s")
            print(f"Speed: {batch_size/fetch_time:.1f} tiles/s")
            
            return images, batch_detections

        return [], []
    def detect(self, interactive=True, force_restart=True):
        """Main detection process with aligned checkpoint and duplicate removal intervals"""
        try:
            start_time = time.time()
            print(f"\n[{datetime.now()}] Starting detection process...")
            
            # Load frame and generate tiles
            frame_gdf = gpd.read_file(self.frame_path)
            tiles = TileGenerator.generate_tiles(
                frame_gdf.total_bounds,
                self.config['tile_size_meters'],
                self.config['tile_overlap']
            )
            total_tiles = len(tiles)
            print(f"Total tiles to process: {total_tiles}")
            
            # Get starting position and previous detections
            if force_restart:
                processed_count = 0
                previous_detections = []
                print("Forced restart: ignoring previous checkpoint.")
            else:
                processed_count, previous_detections = self.checkpoint_manager.load_checkpoint()
            
            all_detections = previous_detections.copy() if previous_detections else []
            
            print(f"Starting from tile: {processed_count + 1}")
            
            # Align intervals
            interval = 2000  # Both checkpoint and duplicate removal use same interval
            last_save = processed_count
            
            with tqdm(
                total=total_tiles,
                initial=processed_count,
                desc="Overall Progress",
                unit="tiles"
            ) as progress_bar:
                while processed_count < total_tiles:
                    batch_end = min(processed_count + self.config['batch_size'], total_tiles)
                    batch_tiles = tiles[processed_count:batch_end]
                    
                    # Process batch
                    images, batch_detections = self._process_batch(
                        batch_tiles,
                        processed_count,
                        total_tiles
                    )
                    
                    if batch_detections:
                        all_detections.extend(batch_detections)
                        
                        # Check if it's time for duplicate removal and checkpoint
                        if processed_count - last_save >= interval:
                            print("\nPerforming duplicate removal...")
                            all_detections = self.results_manager.remove_duplicates(all_detections)
                            
                            print(f"\nSaving checkpoint at {processed_count} tiles...")
                            self.checkpoint_manager.save_checkpoint(
                                processed_count=processed_count,
                                detections=all_detections,
                                total_tiles=total_tiles
                            )
                            last_save = processed_count
                    
                    processed_count = batch_end
                    progress_bar.update(len(batch_tiles))
            
            # Final duplicate removal and results processing
            print("\nPerforming final duplicate removal...")
            all_detections = self.results_manager.remove_duplicates(all_detections)
            return self.results_manager.process_results(all_detections)
            
        except Exception as e:
            print(f"\nError in detection process: {str(e)}")
            return None

        finally:
            if hasattr(self, 'gpu_handler'):
                self.gpu_handler.cleanup()
            if hasattr(self, 'gpu_monitor') and self.gpu_monitor:
                self.gpu_monitor.stop()

    def _print_final_stats(self, results_gdf, start_time):
        """Print final processing statistics"""
        total_time = (time.time() - start_time) / 60
        print("\nFinal Statistics:")
        print(f"- Processing time: {total_time:.1f} minutes")
        print(f"- Total detections: {len(results_gdf)}")
        print(f"- Average confidence: {results_gdf['confidence'].mean():.3f}")
        print(f"- Detection rate: {len(results_gdf) / total_time:.1f} detections/minute")

    def preprocess_image(self, image):
        """Add contrast enhancement for shadowed areas"""
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.2)  # Slightly increase contrast
        return enhanced

    def _initialize_handlers(self):
        """Initialize WMS and GPU handlers"""
        try:
            if not self.wms_handler:
                self.wms_handler = WMSHandler(
                    wms_url=self.config['wms_url'],
                    layer=self.config['wms_layer'],
                    srs=self.config['wms_srs'],
                    size=self.config['wms_size'],
                    image_format=self.config['wms_format'],
                    timeout=self.config.get('timeout', 45),
                    num_workers=self.config['num_workers']
                )

            if not self.gpu_handler:
                self.gpu_handler = GPUHandler(
                    model_path=self.model_path,
                    confidence_threshold=self.config['confidence_threshold'],
                    max_gpu_memory=self.config['max_gpu_memory']
                )
        except Exception as e:
            print(f"Error initializing handlers: {str(e)}")
            raise