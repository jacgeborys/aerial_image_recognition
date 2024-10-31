import onnxruntime as ort
import numpy as np
import cv2
from owslib.wms import WebMapService
import geopandas as gpd
from shapely.geometry import Point
import io
from PIL import Image
import os
import torch
import concurrent.futures
import requests
from tqdm.auto import tqdm
import time
import gc
import math
import json
import psutil
from pyproj import Transformer
from shapely.ops import transform
import warnings
import traceback
from check_gpu import GPUMonitor  # Import the monitor from check_gpu.py


class CarDetector:
    def __init__(self):
        print("Initializing detector...")
        self.base_dir = r"C:\Users\Asus\OneDrive\Pulpit\Rozne\QGIS\car_recognition"

        # Initialize GPU monitor
        self.gpu_monitor = GPUMonitor(log_interval=30)

        # Create directory structure
        self.setup_directories()

        # Paths
        self.model_path = os.path.join(self.base_dir, "models", "car_aerial_detection_yolo7_ITCVD_deepness.onnx")
        self.frame_path = os.path.join(self.base_dir, "gis", "shp", "frames", "warsaw.shp")
        self.output_path = os.path.join(self.base_dir, "gis", "shp", "detection_results", "warsaw.geojson")

        self.config = {
            'wms_url': "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution",
            'tile_size_meters': 64.0,
            'confidence_threshold': 0.4,
            'tile_overlap': 0.1,
            'batch_size': 1024,  # Increased for better speed
            'checkpoint_interval': 1000,  # Less frequent checkpoints
            'num_workers': 16,  # Balanced for your system
            'queue_size': 1024,  # Increased for better GPU utilization
            'max_gpu_memory': 5.0,  # Using more GPU memory
            'min_gpu_memory': 2.0,
            'ram_chunk_size': 512,  # Increased for better performance
            'duplicate_distance': 2.0
        }

        # Initialize checkpoint files and last cleanup time
        self.initialize_checkpoints()
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 300  # Clean up every 5 minutes

        print(f"\nConfiguration:")
        print(f"- GPU target memory: {self.config['min_gpu_memory']}-{self.config['max_gpu_memory']}GB")
        print(f"- Batch size: {self.config['batch_size']}")
        print(f"- Queue size: {self.config['queue_size']}")
        print(f"- Workers: {self.config['num_workers']}")
        print(f"- Checkpoint interval: {self.config['checkpoint_interval']} tiles")

        self._setup_components()

    def setup_directories(self):
        """Create necessary directories and verify their existence"""
        # Create main directories with Windows-style paths
        directories = [
            os.path.join(self.base_dir, "checkpoints"),
            os.path.join(self.base_dir, "models"),
            os.path.join(self.base_dir, "gis", "shp", "detection_results"),
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            if os.path.exists(directory):
                print(f"Verified directory: {directory}")
            else:
                raise RuntimeError(f"Failed to create directory: {directory}")

        self.checkpoint_dir = directories[0]

        # Set checkpoint paths
        self.checkpoint_state = os.path.join(self.checkpoint_dir, "processing_state.json")
        self.checkpoint_data = os.path.join(self.checkpoint_dir, "latest_detections.geojson")

    def initialize_checkpoints(self):
        """Initialize checkpoint files with proper timestamp"""
        try:
            initial_state = {
                'initialized': time.time(),
                'processed_tiles': 0,
                'total_tiles': 0,
                'last_processed_index': -1,
                'timestamp': time.time()  # Add timestamp
            }

            if not os.path.exists(self.checkpoint_state):
                with open(self.checkpoint_state, 'w') as f:
                    json.dump(initial_state, f)

            if not os.path.exists(self.checkpoint_data):
                empty_gdf = gpd.GeoDataFrame(columns=['geometry', 'confidence'], crs="EPSG:4326")
                empty_gdf.to_file(self.checkpoint_data, driver='GeoJSON')

            print(f"\nCheckpoint files verified:")
            print(f"- State: {self.checkpoint_state}")
            print(f"- Data: {self.checkpoint_data}")

        except Exception as e:
            print(f"Error initializing checkpoints: {str(e)}")
            raise

    def _setup_components(self):
        """Initialize processing components"""
        self._setup_gpu()
        self._setup_model()
        self._setup_wms()

    def _setup_gpu(self):
        """Configure GPU and CUDA"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")

        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.fastest = True
        device = torch.cuda.current_device()

        print(f"\nGPU Information:")
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory/1e9:.1f}GB")

        # Warm up GPU
        dummy = torch.zeros(1, 3, 640, 640, device='cuda')
        del dummy
        torch.cuda.empty_cache()

    def _setup_model(self):
        """Initialize ONNX model with GPU optimization"""
        provider_options = {
            'device_id': 0,
            'gpu_mem_limit': int(self.config['max_gpu_memory'] * 1024 * 1024 * 1024),
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True
        }

        providers = [('CUDAExecutionProvider', provider_options)]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )
        print("Model loaded with GPU optimization")

    def _setup_wms(self):
        """Initialize WMS connection"""
        self.wms = WebMapService(self.config['wms_url'], version='1.3.0')
        print("WMS connection established")

    def get_wms_image(self, bbox):
        """Fetch single WMS image with timeout"""
        try:
            img = self.wms.getmap(
                layers=['Raster'],
                srs='EPSG:4326',
                bbox=bbox,
                size=(640, 640),
                format='image/jpeg',
                transparent=False,
                timeout=30  # Add timeout
            )
            return Image.open(io.BytesIO(img.read())).convert('RGB')
        except Exception:
            return None

    def fetch_images_parallel(self, tile_bboxes, progress_bar=None):
        """Fetch images with parallel processing and timeout"""
        results = []
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['num_workers']) as executor:
            # Submit all tasks
            for bbox in tile_bboxes:
                futures.append((executor.submit(self.get_wms_image, bbox), bbox))

            # Process completed tasks
            for future, bbox in futures:
                try:
                    img = future.result(timeout=30)  # Add timeout for each task
                    if img is not None:
                        results.append((img, bbox))
                except (concurrent.futures.TimeoutError, Exception) as e:
                    continue
                finally:
                    if progress_bar:
                        progress_bar.update(1)

        return results

    def preprocess_image(self, img):
        """Preprocess image and move to GPU"""
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        tensor = torch.from_numpy(img_bgr).cuda()
        tensor = tensor.to(dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1) / 255.0
        return tensor

    def process_gpu_queue(self, gpu_queue):
        """Process queue of images in GPU memory"""
        detections = []

        for tensor, bbox in gpu_queue:
            try:
                # Add batch dimension and run inference
                input_tensor = tensor.unsqueeze(0).cpu().numpy()
                outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})

                boxes = outputs[0][0]
                conf_mask = boxes[:, 4] > self.config['confidence_threshold']
                boxes = boxes[conf_mask]

                if len(boxes) > 0:
                    # Move boxes to GPU for faster processing
                    boxes_tensor = torch.from_numpy(boxes).cuda()
                    centers = boxes_tensor[:, :2] / 640

                    lon_offset = bbox[2] - bbox[0]
                    lat_offset = bbox[3] - bbox[1]

                    lons = bbox[0] + (centers[:, 0] * lon_offset)
                    lats = bbox[3] - (centers[:, 1] * lat_offset)
                    confs = boxes_tensor[:, 4]

                    for lon, lat, conf in zip(lons.cpu().numpy(),
                                            lats.cpu().numpy(),
                                            confs.cpu().numpy()):
                        detections.append({
                            'geometry': Point(lon, lat),
                            'confidence': float(conf)
                        })

                    del boxes_tensor

            except Exception:
                continue

        return detections

    def process_tile_batch(self, image_bbox_pairs):
        """Process batch with GPU optimization"""
        if not image_bbox_pairs:
            return []

        all_detections = []
        try:
            # Convert and move all images to GPU first
            gpu_tensors = []
            for img, bbox in image_bbox_pairs:
                try:
                    tensor = self.preprocess_image(img)
                    gpu_tensors.append((tensor, bbox))
                except Exception:
                    continue

            # Process in smaller sub-batches to maintain GPU memory
            for i in range(0, len(gpu_tensors), self.config['queue_size']):
                sub_batch = gpu_tensors[i:i + self.config['queue_size']]
                batch_detections = self.process_gpu_queue(sub_batch)
                all_detections.extend(batch_detections)

            return all_detections

        except Exception as e:
            print(f"Batch processing error: {str(e)}")
            return []
        finally:
            # Clean up GPU memory
            if 'gpu_tensors' in locals():
                del gpu_tensors
            torch.cuda.empty_cache()
            gc.collect()

    def remove_duplicates(self, detections):
        """Remove duplicate detections efficiently"""
        if not detections:
            return []

        try:
            gdf = gpd.GeoDataFrame(detections, crs="EPSG:4326")

            # Convert to UTM for accurate distance calculation
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
            gdf['geometry'] = gdf['geometry'].apply(lambda p: transform(transformer.transform, p))

            # Sort by confidence
            gdf = gdf.sort_values('confidence', ascending=False)

            # Remove duplicates
            kept_indices = []
            for idx in gdf.index:
                if idx in kept_indices:
                    continue

                point = gdf.loc[idx, 'geometry']
                distances = gdf['geometry'].apply(lambda p: point.distance(p))
                duplicates = distances[distances < self.config['duplicate_distance']].index
                kept_indices.extend(duplicates)

            # Convert back to WGS84
            transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
            filtered_gdf = gdf.loc[kept_indices]
            filtered_gdf['geometry'] = filtered_gdf['geometry'].apply(
                lambda p: transform(transformer.transform, p)
            )

            return filtered_gdf.to_dict('records')

        except Exception as e:
            print(f"Deduplication error: {str(e)}")
            return detections  # Return original if deduplication fails

    def save_checkpoint(self, detections, processed_tiles, total_tiles):
        """Save checkpoint with processing state"""
        try:
            # Save processing state
            checkpoint_data = {
                'processed_tiles': processed_tiles,
                'total_tiles': total_tiles,
                'last_processed_index': processed_tiles - 1,
                'timestamp': time.time()
            }

            with open(self.checkpoint_state, 'w') as f:
                json.dump(checkpoint_data, f)

            # Save detections
            unique_detections = self.remove_duplicates(detections)
            gdf = gpd.GeoDataFrame(unique_detections, crs="EPSG:4326")
            gdf.to_file(self.checkpoint_data, driver='GeoJSON')

            print(f"\nCheckpoint saved at {processed_tiles}/{total_tiles} tiles")
            print(f"Stored {len(unique_detections)} unique detections")

        except Exception as e:
            print(f"Checkpoint save error: {str(e)}")

    def load_checkpoint(self):
        """Load checkpoint with proper error handling"""
        if os.path.exists(self.checkpoint_state) and os.path.exists(self.checkpoint_data):
            try:
                with open(self.checkpoint_state, 'r') as f:
                    checkpoint_data = json.load(f)

                # Ensure all required fields exist
                if 'timestamp' not in checkpoint_data:
                    checkpoint_data['timestamp'] = time.time()
                if 'processed_tiles' not in checkpoint_data:
                    checkpoint_data['processed_tiles'] = 0
                if 'total_tiles' not in checkpoint_data:
                    checkpoint_data['total_tiles'] = 0
                if 'last_processed_index' not in checkpoint_data:
                    checkpoint_data['last_processed_index'] = -1

                gdf = gpd.read_file(self.checkpoint_data)
                detections = gdf.to_dict('records') if not gdf.empty else []

                print(f"\nFound checkpoint:")
                print(f"- Processed: {checkpoint_data['processed_tiles']} tiles")
                print(f"- Detections: {len(detections)}")
                print(f"- Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(checkpoint_data['timestamp']))}")

                response = input("\nResume from checkpoint? (y/n): ").lower()
                if response == 'y':
                    return checkpoint_data, detections

                print("Starting fresh processing...")
                self.initialize_checkpoints()

            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                print("Starting fresh processing...")
                self.initialize_checkpoints()

        return {'processed_tiles': 0, 'total_tiles': 0, 'last_processed_index': -1, 'timestamp': time.time()}, []

    def generate_tiles(self, bounds):
        """Generate tile coordinates"""
        minx, miny, maxx, maxy = bounds
        mid_lat = (miny + maxy) / 2

        # Convert meters to degrees
        earth_radius = 6378137
        tile_meters = self.config['tile_size_meters']
        lat_deg = tile_meters / (earth_radius * math.pi / 180)
        lon_deg = tile_meters / (earth_radius * math.pi / 180 * math.cos(math.radians(mid_lat)))

        overlap = self.config['tile_overlap']
        step_lon = lon_deg * (1 - overlap)
        step_lat = lat_deg * (1 - overlap)

        tiles = []
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                tiles.append((
                    x, y,
                    min(x + lon_deg, maxx),
                    min(y + lat_deg, maxy)
                ))
                y += step_lat
            x += step_lon

        return tiles

    def detect(self):
        """Main detection process with optimized performance"""
        try:
            # Start GPU monitoring
            self.gpu_monitor.start()
            start_time = time.time()

            # Load frame
            frame_gdf = gpd.read_file(self.frame_path)
            if frame_gdf.crs.to_epsg() != 4326:
                frame_gdf = frame_gdf.to_crs(epsg=4326)

            # Generate tiles
            tiles = self.generate_tiles(frame_gdf.total_bounds)
            total_tiles = len(tiles)
            del frame_gdf
            gc.collect()

            # Load checkpoint or initialize
            checkpoint_data, all_detections = self.load_checkpoint()
            if checkpoint_data:
                processed_count = checkpoint_data['processed_tiles']
                start_idx = checkpoint_data['last_processed_index'] + 1
            else:
                processed_count = 0
                start_idx = 0
                all_detections = []

            print(f"\nProcessing {total_tiles} tiles...")
            progress_bar = tqdm(
                total=total_tiles,
                initial=processed_count,
                desc="Processing",
                unit="tiles",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
            )
            progress_bar.set_postfix({"detections": len(all_detections)})

            last_checkpoint = processed_count

            # Process tiles in batches
            for idx in range(start_idx, total_tiles, self.config['batch_size']):
                try:
                    # Get current batch
                    end_idx = min(idx + self.config['batch_size'], total_tiles)
                    current_batch = tiles[idx:end_idx]

                    # Process batch and update progress
                    image_bbox_pairs = self.fetch_images_parallel(current_batch, progress_bar)

                    if image_bbox_pairs:
                        batch_detections = self.process_tile_batch(image_bbox_pairs)

                        # Manage memory for detections
                        if len(all_detections) > 100000:
                            all_detections = self.remove_duplicates(all_detections)

                        all_detections.extend(batch_detections)
                        progress_bar.set_postfix({"detections": len(all_detections)})

                        # Check for checkpoint
                        if processed_count - last_checkpoint >= self.config['checkpoint_interval']:
                            self.save_checkpoint(all_detections, processed_count, total_tiles)
                            last_checkpoint = processed_count

                    # Clean up
                    del image_bbox_pairs
                    gc.collect()
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\nBatch error: {str(e)}")
                    self.save_checkpoint(all_detections, processed_count, total_tiles)
                    continue

            progress_bar.close()

            # Continuing from detect() method
            # Process final results
            if all_detections:
                print("\nProcessing final results...")
                final_detections = self.remove_duplicates(all_detections)
                results_gdf = gpd.GeoDataFrame(final_detections, crs="EPSG:4326")

                print(f"Saving {len(results_gdf)} final detections...")
                results_gdf.to_file(self.output_path, driver='GeoJSON')

                total_time = (time.time() - start_time) / 60
                print(f"\nProcessing Complete!")
                print(f"Time: {total_time:.1f} minutes")
                print(f"Tiles: {total_tiles}")
                print(f"Detections: {len(results_gdf)}")
                print(f"Results saved to: {self.output_path}")

                # Clean up checkpoint files on successful completion
                if os.path.exists(self.checkpoint_state):
                    os.remove(self.checkpoint_state)
                if os.path.exists(self.checkpoint_data):
                    os.remove(self.checkpoint_data)

                return results_gdf

            return gpd.GeoDataFrame(columns=['geometry', 'confidence'], crs="EPSG:4326")

        except Exception as e:
            print(f"\nError in detection process: {str(e)}")
            traceback.print_exc()
            # Try to save checkpoint on error
            try:
                self.save_checkpoint(all_detections, processed_count, total_tiles)
                print("Checkpoint saved after error - you can resume later")
            except:
                print("Failed to save checkpoint after error")
            return gpd.GeoDataFrame(columns=['geometry', 'confidence'], crs="EPSG:4326")
        finally:
            # Final cleanup
            torch.cuda.empty_cache()
            gc.collect()

            self.gpu_monitor.stop()

def main():
    """Main execution with error handling"""
    try:
        # Initialize detector
        detector = CarDetector()

        # Run detection
        print("\nStarting detection process...")
        start_time = time.time()

        results = detector.detect()

        if len(results) > 0:
            print(f"\nFinal Statistics:")
            print(f"- Total detections: {len(results)}")
            print(f"- Average confidence: {results['confidence'].mean():.3f}")

            # Memory usage summary
            ram_usage = psutil.Process().memory_info().rss / 1e9
            gpu_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_reserved = torch.cuda.memory_reserved() / 1e9

            print(f"\nFinal Memory Usage:")
            print(f"- RAM: {ram_usage:.1f}GB")
            print(f"- GPU Allocated: {gpu_allocated:.1f}GB")
            print(f"- GPU Reserved: {gpu_reserved:.1f}GB")

            total_time = (time.time() - start_time) / 60
            print(f"\nTotal processing time: {total_time:.1f} minutes")
            print(f"Average speed: {len(results)/total_time:.1f} detections/minute")

            return results
        else:
            print("No detections found")
            return None

    except Exception as e:
        print(f"Error in main process: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()