import os
import time
from tqdm import tqdm
import geopandas as gpd
import gc
import traceback
import torch

from .wms_handler import WMSHandler
from .gpu_handler import GPUHandler
from .utils import TileGenerator, CheckpointManager, ResultsManager
from .monitors import GPUMonitor


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
        print("\nInitializing WMS connection and GPU...")
        self.wms_handler = WMSHandler(
            self.config['wms_url'],
            num_workers=self.config['num_workers']
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
            'tile_size_meters': 64.0,
            'confidence_threshold': 0.4,
            'tile_overlap': 0.1,
            'batch_size': 1024,
            'checkpoint_interval': 1000,
            'num_workers': 16,
            'queue_size': 1024,
            'max_gpu_memory': 5.0,
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
        """Initialize WMS and GPU handlers"""
        if not self.wms_handler:
            self.wms_handler = WMSHandler(
                self.config['wms_url'],
                num_workers=self.config['num_workers']
            )

        if not self.gpu_handler:
            self.gpu_handler = GPUHandler(
                self.model_path,
                max_gpu_memory=self.config['max_gpu_memory'],
                confidence_threshold=self.config['confidence_threshold']
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

    def _process_batch(self, batch_tiles):
        """Process a single batch with clear status reporting"""
        batch_size = len(batch_tiles)

        # 1. Fetch images
        print("\n" + "=" * 50)
        print(f"FETCHING IMAGES: Batch of {batch_size} tiles")
        print("=" * 50)

        with tqdm(
            total=batch_size,
            desc="Downloading",
            position=1,
            leave=False
        ) as fetch_progress:
            images = self.wms_handler.fetch_batch(batch_tiles, fetch_progress)

        print(f"\nFetch complete: {len(images)}/{batch_size} images downloaded")

        # 2. Process through YOLO
        if images:
            print("\n" + "=" * 50)
            print("RUNNING YOLO DETECTION")
            print("=" * 50)

            # Use queue_size from config, defaulting to batch_size if not specified
            queue_size = self.config.get('queue_size', self.config['batch_size'])
            batch_detections = self.gpu_handler.process_batch(
                images,
                queue_size=queue_size
            )
            print(f"Detection complete: Found {len(batch_detections)} cars")
            return images, batch_detections

        return [], []
    def detect(self, interactive=True):
        """Main detection process with clear progress reporting"""
        try:
            tqdm.write("\nStarting detection process...")
            
            # Load frame and generate tiles
            frame_gdf = gpd.read_file(self.frame_path)
            tiles = TileGenerator.generate_tiles(
                frame_gdf.total_bounds,
                self.config['tile_size_meters'],
                self.config['tile_overlap']
            )
            total_tiles = len(tiles)
            
            # Generate and save tile preview
            tqdm.write("\nGenerating tile preview...")
            frame_name = os.path.splitext(os.path.basename(self.frame_path))[0]
            self.wms_handler.preview_tiles(
                tiles, 
                self.preview_dir, 
                prefix=frame_name
            )
            
            # Get starting position and previous detections from checkpoint
            start_idx, previous_detections = self.checkpoint_manager.load_checkpoint()
            processed_count = start_idx
            all_detections = previous_detections.copy()  # Start with previous detections
            last_checkpoint = processed_count
            
            if start_idx > 0:
                tqdm.write(f"\nResuming from checkpoint at tile {start_idx} of {total_tiles}")

            # Initialize GPU monitor
            self.gpu_monitor = GPUMonitor(log_interval=30)
            self.gpu_monitor.start()
            start_time = time.time()

            # Create main progress bar
            main_progress = tqdm(
                total=total_tiles,
                initial=processed_count,
                desc="Overall Progress",
                position=0,
                leave=True
            )

            # Process tiles in batches
            for idx in range(start_idx, total_tiles, self.config['batch_size']):
                batch_tiles = tiles[idx:min(idx + self.config['batch_size'], total_tiles)]
                batch_size = len(batch_tiles)

                if interactive:
                    input("\nPress Enter to start processing next batch...")

                # Status update using tqdm.write
                tqdm.write(f"\nBatch Status:")
                tqdm.write(f"- Processing tiles {idx} to {idx + batch_size} of {total_tiles}")
                tqdm.write(f"- Current detections: {len(all_detections)}")
                tqdm.write(f"- GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB used")

                # Process images with proper progress bar handling
                with tqdm(
                    total=batch_size,
                    desc="Fetching Images",
                    position=1,
                    leave=False,
                    ncols=100
                ) as batch_progress:
                    # Use parallel fetching
                    tqdm.write("\nFetching images from WMS...")
                    images = self.wms_handler.fetch_batch(batch_tiles, batch_progress)

                tqdm.write(f"Retrieved {len(images)}/{batch_size} images successfully")

                if images:
                    # Process images
                    tqdm.write("\nProcessing images through YOLO...")
                    batch_detections = self.process_images(images)
                    tqdm.write(f"Found {len(batch_detections)} detections in this batch")

                    # Update progress and save results
                    if batch_detections:
                        all_detections.extend(batch_detections)
                        if len(all_detections) > 1000:
                            tqdm.write("\nRemoving duplicate detections...")
                            all_detections = self.results_manager.remove_duplicates(all_detections)
                            tqdm.write(f"Unique detections: {len(all_detections)}")

                    processed_count += batch_size
                    main_progress.update(batch_size)
                    main_progress.set_postfix({
                        "detections": len(all_detections),
                        "batch": f"{len(batch_detections)} found"
                    })

                    # Save checkpoint
                    if processed_count - last_checkpoint >= self.config['checkpoint_interval']:
                        tqdm.write("\nSaving checkpoint...")
                        self.checkpoint_manager.save_checkpoint(
                            all_detections, processed_count, total_tiles
                        )
                        last_checkpoint = processed_count

                # Cleanup
                del images
                gc.collect()
                torch.cuda.empty_cache()

            main_progress.close()

            # Process final results
            tqdm.write("\nProcessing complete! Saving final results...")
            final_detections = self.results_manager.remove_duplicates(all_detections)
            results_gdf = self.checkpoint_manager._create_geodataframe(final_detections)
            results_gdf.to_file(self.output_path, driver='GeoJSON')

            self._print_final_stats(results_gdf, start_time)
            return results_gdf

        except Exception as e:
            tqdm.write(f"\nError in detection process: {str(e)}")
            traceback.print_exc()
            if 'all_detections' in locals() and 'processed_count' in locals():
                tqdm.write("\nSaving checkpoint after error...")
                self.checkpoint_manager.save_checkpoint(
                    all_detections, processed_count, total_tiles
                )
            return None

        finally:
            if hasattr(self, 'gpu_handler'):
                self.gpu_handler.cleanup()
            self.gpu_monitor.stop()

    def _print_final_stats(self, results_gdf, start_time):
        """Print final processing statistics"""
        total_time = (time.time() - start_time) / 60
        print("\nFinal Statistics:")
        print(f"- Processing time: {total_time:.1f} minutes")
        print(f"- Total detections: {len(results_gdf)}")
        print(f"- Average confidence: {results_gdf['confidence'].mean():.3f}")
        print(f"- Detection rate: {len(results_gdf) / total_time:.1f} detections/minute")