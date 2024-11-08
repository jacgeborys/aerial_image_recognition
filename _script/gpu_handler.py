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
        """Process image with multiple contrast/brightness variations and occlusion handling"""
        variations = []
        
        # Original image processing (keep existing variations)
        variations.extend(self._get_lighting_variations(img))
        
        # Add occlusion-focused variations
        variations.extend(self._get_occlusion_variations(img))
        
        return variations

    def _get_lighting_variations(self, img):
        """Get variations for different lighting conditions with enhanced shadow handling"""
        variations = []
        
        # Original image
        variations.append(self._prepare_tensor(img))
        
        # More aggressive shadow enhancement
        enhancer = ImageEnhance.Brightness(img)
        very_bright_img = enhancer.enhance(1.8)  # Increased from 1.3
        variations.append(self._prepare_tensor(very_bright_img))
        
        # Multi-step shadow enhancement
        shadow_img = img
        for brightness in [1.4, 1.6]:  # Multiple brightness levels
            shadow_img = ImageEnhance.Brightness(shadow_img).enhance(brightness)
            shadow_img = ImageEnhance.Contrast(shadow_img).enhance(1.3)
            variations.append(self._prepare_tensor(shadow_img))
        
        # Gamma correction for shadow areas
        img_array = np.array(img)
        gamma = 1.5  # Adjust gamma to brighten shadows while preserving highlights
        gamma_corrected = np.power(img_array / 255.0, 1.0 / gamma) * 255.0
        gamma_corrected = gamma_corrected.astype(np.uint8)
        variations.append(self._prepare_tensor(Image.fromarray(gamma_corrected)))
        
        return variations

    def _get_occlusion_variations(self, img):
        """Get variations to handle occlusions with enhanced shadow detection"""
        variations = []
        img_array = np.array(img)
        
        # Enhanced shadow removal using CLAHE
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE multiple times with different parameters
        clahe_variations = [
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)),
            cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4)),  # More aggressive
            cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))  # Larger tiles
        ]
        
        for clahe in clahe_variations:
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            shadow_removed = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            variations.append(self._prepare_tensor(Image.fromarray(shadow_removed)))
        
        # Rest of the existing variations...
        # (Keep the existing adaptive thresholding, edge enhancement, and color-based segmentation)
        
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
        """Run inference on tensor batch with variations and confidence adjustment"""
        detections = []
        try:
            for tensor_variations, bbox in tensor_batch:
                batch_boxes = []
                
                # Process each variation
                for i, tensor in enumerate(tensor_variations):
                    input_tensor = tensor.unsqueeze(0).cpu().numpy()
                    outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
                    boxes = outputs[0][0]
                    
                    # Adjust confidence based on variation type
                    conf_adjustment = self._get_confidence_adjustment(i, len(tensor_variations))
                    boxes[:, 4] *= conf_adjustment
                    
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

    def _get_confidence_adjustment(self, variation_index, total_variations):
        """Adjust confidence scores with enhanced shadow detection weights"""
        # Original lighting variations
        if variation_index < 5:  # Now we have more lighting variations
            return 1.0
        
        # Shadow-specific variations should have higher confidence
        shadow_adjustments = {
            5: 0.98,  # Aggressive brightness
            6: 0.98,  # Multi-step shadow 1
            7: 0.98,  # Multi-step shadow 2
            8: 0.95,  # Gamma correction
            9: 0.95,  # CLAHE variation 1
            10: 0.95, # CLAHE variation 2
            11: 0.95  # CLAHE variation 3
        }
        
        return shadow_adjustments.get(variation_index, 0.85)

    def cleanup(self):
        """Clean up GPU resources"""
        torch.cuda.empty_cache()
        gc.collect()