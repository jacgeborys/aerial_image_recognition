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


class CarDetector:
    def __init__(self, base_dir, config=None):
        """Initialize detector with configuration"""
        try:
            print("\nInitializing detector...")
            self.base_dir = base_dir
            self.config = self._load_config(config)
            
            # Set up paths
            self.frame_path = os.path.join(base_dir, 'gis', 'frames', self.config['frame_path'])
            self.output_dir = os.path.join(base_dir, 'output')
            self.model_path = os.path.join(base_dir, 'models', self.config['model_path'])
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Initialize components
            print("\nInitializing WMS connection and GPU...")
            self.wms_handler = WMSHandler(
                wms_url=self.config['wms_url'],
                num_workers=self.config['num_workers']
            )
            self.gpu_handler = GPUHandler(
                model_path=self.model_path,
                confidence_threshold=self.config['confidence_threshold'],
                max_gpu_memory=self.config['max_gpu_memory']
            )
            self.checkpoint_manager = CheckpointManager(self.output_dir)
            self.results_manager = ResultsManager(self.output_dir)
            
        except Exception as e:
            print(f"Error initializing detector: {str(e)}")
            raise

    def _load_config(self, custom_config=None):
        """Load configuration with custom overrides"""
        config = self._get_default_config()
        if custom_config:
            config.update(custom_config)
        return config

    def _get_default_config(self):
        """Get default configuration"""
        return {
            'wms_url': "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution",
            'tile_size_meters': 64.0,
            'confidence_threshold': 0.4,
            'tile_overlap': 0.1,
            'batch_size': 64,
            'checkpoint_interval': 50,
            'max_gpu_memory': 5.0,
            'duplicate_distance': 1.0,
            'frame_path': 'warsaw_central.shp',
            'model_path': 'car_aerial_detection_yolo7_ITCVD_deepness.onnx'
        }

    def _setup_paths(self):
        """Setup and create necessary directories"""
        # 1. Base directories
        self.base_dir = os.path.abspath(self.base_dir)
        
        # 2. Main directory structure
        self.gis_dir = os.path.join(self.base_dir, "gis")
        self.model_dir = os.path.join(self.base_dir, "models")
        
        # 3. GIS subdirectories
        self.frame_dir = os.path.join(self.gis_dir, "frames")
        self.output_dir = os.path.join(self.gis_dir, "detection_results")
        self.checkpoint_dir = os.path.join(self.gis_dir, "checkpoints")
        self.preview_dir = os.path.join(self.gis_dir, "preview_tiles")
        
        # 4. Create all directories
        directories = [
            self.gis_dir,
            self.model_dir,
            self.frame_dir,
            self.output_dir,
            self.checkpoint_dir,
            self.preview_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # 5. Set input/output file paths
        frame_name = self.config.get('frame_path', 'warsaw_central.shp')
        model_name = self.config.get('model_path', 'car_aerial_detection_yolo7_ITCVD_deepness.onnx')
        
        # Remove file extension for output naming
        base_frame_name = os.path.splitext(frame_name)[0]
        
        self.frame_path = os.path.join(self.frame_dir, frame_name)
        self.model_path = os.path.join(self.model_dir, model_name)
        self.output_path = os.path.join(self.output_dir, f"{base_frame_name}.geojson")
        
        # 6. Print paths for debugging
        if self.config.get('debug', False):
            print("\nDirectory Structure:")
            print(f"Base Dir: {self.base_dir}")
            print(f"GIS Dir: {self.gis_dir}")
            print(f"Model Dir: {self.model_dir}")
            print(f"Frame Dir: {self.frame_dir}")
            print(f"Output Dir: {self.output_dir}")
            print(f"Checkpoint Dir: {self.checkpoint_dir}")
            print(f"Preview Dir: {self.preview_dir}")
            print(f"\nFile Paths:")
            print(f"Frame: {self.frame_path}")
            print(f"Model: {self.model_path}")
            print(f"Output: {self.output_path}")

    def _print_config(self):
        """Print current configuration"""
        print("\nConfiguration:")
        print(f"- Tile size: {self.config['tile_size_meters']}m")
        print(f"- Batch size: {self.config['batch_size']}")
        print(f"- GPU memory limit: {self.config['max_gpu_memory']}GB")
        print(f"- Confidence threshold: {self.config['confidence_threshold']}")

    def _initialize_handlers(self):
        """Initialize WMS and GPU handlers"""
        try:
            if not self.wms_handler:
                # Initialize WMS handler with only the required parameters
                wms_url = self.config.get('wms_url', 'https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution')
                self.wms_handler = WMSHandler(wms_url)  # Only pass the URL

            if not self.gpu_handler:
                self.gpu_handler = GPUHandler(
                    model_path=self.model_path,
                    confidence_threshold=self.config.get('confidence_threshold', 0.3),
                    max_gpu_memory=self.config.get('max_gpu_memory', 4.0)
                )
        except Exception as e:
            print(f"Error initializing handlers: {str(e)}")
            raise

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
    def detect(self, interactive=True):
        """Main detection process with clear progress reporting"""
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
            processed_count, previous_detections = self.checkpoint_manager.load_checkpoint()
            all_detections = previous_detections.copy() if previous_detections else []
            
            # Process in batches
            batch_size = self.config['batch_size']
            with tqdm(
                total=total_tiles,
                initial=processed_count,
                desc="Overall Progress",
                unit="tiles"
            ) as progress_bar:
                while processed_count < total_tiles:
                    # Get next batch
                    batch_end = min(processed_count + batch_size, total_tiles)
                    batch_tiles = tiles[processed_count:batch_end]
                    
                    # Process batch
                    images, batch_detections = self._process_batch(
                        batch_tiles,
                        processed_count,
                        total_tiles
                    )
                    
                    # Update progress and save results
                    if batch_detections:
                        all_detections.extend(batch_detections)
                        # Save checkpoint with processed count and detections
                        self.checkpoint_manager.save_checkpoint(
                            processed_count=batch_end,
                            detections=all_detections,
                            total_tiles=total_tiles
                        )
                    
                    processed_count = batch_end
                    progress_bar.update(len(batch_tiles))
            
            # Final results processing
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