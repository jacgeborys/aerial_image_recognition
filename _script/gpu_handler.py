import torch
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import gc
import tqdm


class GPUHandler:
    def __init__(self, model_path, max_gpu_memory=5.0, confidence_threshold=0.4):
        self.model_path = model_path
        self.max_gpu_memory = max_gpu_memory
        self.confidence_threshold = confidence_threshold
        self.session = None
        self._setup_gpu()
        self._load_model()

    def _setup_gpu(self):
        """Configure GPU settings"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")

        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.fastest = True

        # Warm up GPU
        dummy = torch.zeros(1, 3, 640, 640, device='cuda')
        del dummy
        torch.cuda.empty_cache()

    def _load_model(self):
        """Initialize ONNX model with GPU optimization"""
        provider_options = {
            'device_id': 0,
            'gpu_mem_limit': int(self.max_gpu_memory * 1024 * 1024 * 1024),
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

    def preprocess_image(self, img):
        """Process image with multiple contrast/brightness variations"""
        variations = []
        
        # Original image
        variations.append(self._prepare_tensor(img))
        
        # Enhance shadows (increase brightness)
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(1.3)
        variations.append(self._prepare_tensor(bright_img))
        
        # Enhance contrast for better edge detection
        enhancer = ImageEnhance.Contrast(img)
        contrast_img = enhancer.enhance(1.4)
        variations.append(self._prepare_tensor(contrast_img))
        
        # Enhance dark areas
        enhancer = ImageEnhance.Brightness(img)
        dark_img = enhancer.enhance(0.7)
        contrast_enhancer = ImageEnhance.Contrast(dark_img)
        dark_contrast_img = contrast_enhancer.enhance(1.5)
        variations.append(self._prepare_tensor(dark_contrast_img))
        
        return variations

    def _prepare_tensor(self, img):
        """Convert single image to tensor"""
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        tensor = torch.from_numpy(img_bgr).cuda()
        tensor = tensor.to(dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1) / 255.0
        return tensor

    def process_batch(self, image_bbox_pairs, queue_size=512):
        """Process a batch of images with YOLO"""
        if not image_bbox_pairs:
            return []

        print("\nPreparing images for GPU processing...")
        all_detections = []
        try:
            # Preallocate GPU memory
            gpu_tensors = []
            for img, bbox in image_bbox_pairs:
                try:
                    tensor = self.preprocess_image(img)
                    gpu_tensors.append((tensor, bbox))
                except Exception:
                    continue

            # Process in smaller sub-batches
            total_sub_batches = (len(gpu_tensors) + queue_size - 1) // queue_size
            print(f"Processing {len(gpu_tensors)} images in {total_sub_batches} sub-batches...")

            for i in range(0, len(gpu_tensors), queue_size):
                sub_batch = gpu_tensors[i:i + queue_size]
                batch_detections = self._process_tensors(sub_batch)
                all_detections.extend(batch_detections)
                
                # Force memory cleanup after each sub-batch
                del sub_batch
                torch.cuda.empty_cache()

            return all_detections

        except Exception as e:
            print(f"GPU processing error: {str(e)}")
            return []
        finally:
            if 'gpu_tensors' in locals():
                del gpu_tensors
            torch.cuda.empty_cache()
            gc.collect()

    def _process_tensors(self, tensor_batch):
        """Run inference on tensor batch with variations"""
        detections = []
        try:
            for tensor_variations, bbox in tensor_batch:
                batch_boxes = []
                
                # Process each variation
                for tensor in tensor_variations:
                    input_tensor = tensor.unsqueeze(0).cpu().numpy()
                    outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
                    boxes = outputs[0][0]
                    
                    # Apply confidence threshold
                    conf_mask = boxes[:, 4] > self.confidence_threshold
                    filtered_boxes = boxes[conf_mask]
                    
                    if len(filtered_boxes) > 0:
                        batch_boxes.append(filtered_boxes)
                
                # Combine detections from all variations
                if batch_boxes:
                    combined_boxes = np.concatenate(batch_boxes, axis=0)
                    boxes_tensor = torch.from_numpy(combined_boxes).cuda()
                    
                    # Convert to coordinates
                    centers = boxes_tensor[:, :2] / 640
                    lon_offset = bbox[2] - bbox[0]
                    lat_offset = bbox[3] - bbox[1]
                    
                    lons = bbox[0] + (centers[:, 0] * lon_offset)
                    lats = bbox[3] - (centers[:, 1] * lat_offset)
                    confs = boxes_tensor[:, 4]
                    
                    # Add all detections
                    for lon, lat, conf in zip(lons.cpu().numpy(),
                                            lats.cpu().numpy(),
                                            confs.cpu().numpy()):
                        detections.append({
                            'lon': lon,
                            'lat': lat,
                            'confidence': float(conf)
                        })
                    
                    del boxes_tensor
                    
        except Exception as e:
            print(f"Error processing tensor batch: {str(e)}")
        
        return detections

    def cleanup(self):
        """Clean up GPU resources"""
        torch.cuda.empty_cache()
        gc.collect()