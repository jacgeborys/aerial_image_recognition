import os
import time
from tqdm import tqdm
import geopandas as gpd
import gc
import traceback
import torch
from PIL import ImageEnhance, ImageDraw
from datetime import datetime

from .wms_handler import WMSHandler
from .gpu_handler import GPUHandler
from .utils import TileGenerator, CheckpointManager, ResultsManager
from .monitors import GPUMonitor
from .wmts_handler import WMTSHandler


class CarDetector:
    def __init__(self, base_dir, config=None):
        """Initialize detector with configuration"""
        print("Initializing detector...")
        self.base_dir = base_dir
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # Setup paths
        self._setup_paths()

        # Extract frame name for checkpoint files
        self.frame_name = os.path.splitext(os.path.basename(self.frame_path))[0]

        # Initialize components
        self.gpu_monitor = GPUMonitor(log_interval=30)
        self.checkpoint_manager = CheckpointManager(
            self.checkpoint_dir, 
            prefix=self.frame_name
        )
        self.results_manager = ResultsManager(duplicate_distance=self.config['duplicate_distance'])

        # Initialize handlers immediately
        print("\nInitializing WMTS connection and GPU...")
        self.wmts_handler = WMTSHandler(
            num_workers=self.config['num_workers'],
            timeout=30
        )
        self.gpu_handler = GPUHandler(
            self.model_path,
            max_gpu_memory=self.config['max_gpu_memory'],
            confidence_threshold=self.config['confidence_threshold']
        )

        self._print_config()
        self.gpu_update_interval = 5
        self.last_gpu_update = 0

    def _get_default_config(self):
        """Get default configuration"""
        return {
            'wms_url': "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution",
            'tile_size_meters': 32.0,
            'confidence_threshold': 0.4,
            'tile_overlap': 0.1,
            'batch_size': 1024,
            'checkpoint_interval': 1000,
            'num_workers': 16,
            'queue_size': 1024,
            'max_gpu_memory': 5.5,
            'duplicate_distance': 1.0,
            'frame_path': 'warsaw_central.shp',  # Default frame file name
            'model_path': 'car_aerial_detection_yolo7_ITCVD_deepness.onnx'  # Default model file name
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
        print(f"- Workers: {self.config['num_workers']}")
        print(f"- GPU memory limit: {self.config['max_gpu_memory']}GB")
        print(f"- Confidence threshold: {self.config['confidence_threshold']}")

    def _initialize_handlers(self):
        """Initialize handlers"""
        print("\nInitializing WMTS connection and GPU...")
        
        # Initialize WMTS handler
        self.wmts_handler = WMTSHandler(
            num_workers=self.config['num_workers'],
            timeout=30
        )
        
        # Initialize GPU handler
        self.gpu_handler = GPUHandler(
            model_path=os.path.join(self.model_dir, "model.onnx"),
            confidence_threshold=self.config['confidence_threshold'],
            max_memory_gb=self.config['max_gpu_memory']
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
        """Fetch a batch of images from WMTS"""
        try:
            # Ensure WMTS handler is initialized
            if not self.wmts_handler:
                self._initialize_handlers()

            # Fetch images using WMTS
            return self.wmts_handler.fetch_batch(tile_batch, progress_bar)
        except Exception as e:
            print(f"\nError fetching images: {str(e)}")
            return []

    def _process_batch(self, batch_tiles, batch_idx, batch_progress=None):
        """Process a batch of tiles"""
        try:
            batch_start = time.time()
            images_and_bboxes = []
            processed_count = 0
            
            # Process each tile
            for tile_idx, bbox in enumerate(batch_tiles):
                try:
                    # Use WMTS handler instead of WMS
                    img, actual_bbox = self.wmts_handler.get_tile_for_bbox(bbox)
                    if img is not None:
                        images_and_bboxes.append((img, actual_bbox))
                        processed_count += 1
                    
                    if batch_progress:
                        batch_progress.update(1)
                        
                except Exception as e:
                    print(f"Error processing tile {tile_idx}: {str(e)}")
                    continue
            
            batch_time = time.time() - batch_start
            if batch_time > 0:  # Prevent division by zero
                speed = processed_count / batch_time
                print(f"Successfully processed {processed_count}/{len(batch_tiles)} tiles in batch {batch_idx}")
                print(f"Batch speed: {speed:.1f} tiles/second")
            
            # Process detections if we have images
            if images_and_bboxes:
                images = [img for img, _ in images_and_bboxes]
                bboxes = [bbox for _, bbox in images_and_bboxes]
                detections = self.process_images(images, bboxes)
                return detections
            
            return []
            
        except Exception as e:
            print(f"Batch processing error: {str(e)}")
            return []

    def detect(self, interactive=True):
        """Main detection process with clear progress reporting"""
        try:
            start_time = time.time()
            tqdm.write(f"\n[{datetime.now()}] Starting detection process...")
            
            # Load frame and generate tiles
            load_start = time.time()
            frame_gdf = gpd.read_file(self.frame_path)
            tiles = TileGenerator.generate_tiles(
                frame_gdf.total_bounds,
                self.config['tile_size_meters'],
                self.config['tile_overlap']
            )
            total_tiles = len(tiles)
            tqdm.write(f"[{datetime.now()}] Frame loaded and tiles generated in {time.time() - load_start:.1f}s")
            tqdm.write(f"Total tiles to process: {total_tiles}")
            
            # Initialize results list
            all_detections = []
            
            # Process tiles in batches
            for idx in range(0, total_tiles, self.config['batch_size']):
                batch_start = time.time()
                batch_tiles = tiles[idx:min(idx + self.config['batch_size'], total_tiles)]
                
                batch_num = idx//self.config['batch_size'] + 1
                tqdm.write(f"\n[{datetime.now()}] Starting batch {batch_num}")
                
                # Process batch
                batch_detections = self._process_batch(
                    batch_tiles, 
                    batch_num
                )
                
                # Extend detections list
                if batch_detections:
                    all_detections.extend(batch_detections)
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(all_detections, idx + len(batch_tiles), total_tiles)
            
            tqdm.write("\nProcessing complete!")
            
            # Return results if we have any
            if all_detections:
                return gpd.GeoDataFrame(all_detections)
            return None
            
        except Exception as e:
            print(f"Error in detection process: {str(e)}")
            traceback.print_exc()
            return None

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

    def verify_tile_location(self, bbox, img, save_path):
        """Save a tile with its coordinates for verification"""
        try:
            # Create a copy of the image to draw on
            img_draw = img.copy()
            draw = ImageDraw.Draw(img_draw)
            
            # Add coordinates and bbox info
            text = f"Bbox: {bbox}\nCenter: {(bbox[0]+bbox[2])/2:.6f}, {(bbox[1]+bbox[3])/2:.6f}"
            draw.text((10, 10), text, fill='red')
            
            # Save the annotated image
            img_draw.save(save_path)
            return True
        except Exception as e:
            print(f"Error saving verification tile: {str(e)}")
            return False