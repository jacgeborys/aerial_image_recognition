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
            'do_copy_in_default_stream': True,
            'cudnn_conv_use_max_workspace': True
        }

        providers = [('CUDAExecutionProvider', provider_options)]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.intra_op_num_threads = 8
        sess_options.inter_op_num_threads = 8
        sess_options.enable_profiling = False
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

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
        """Get optimized variations for shadow and strong sunlight conditions"""
        variations = []
        img_array = np.array(img)
        
        # 1. Original image
        variations.append(self._prepare_tensor(img))
        
        # 2. Shadow-specific CLAHE enhancement
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        shadow_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        variations.append(self._prepare_tensor(Image.fromarray(shadow_enhanced)))
        
        # 3. Strong brightness for deep shadows
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(2.0)
        variations.append(self._prepare_tensor(bright_img))
        
        # 4. Gamma correction for shadow details
        gamma = 2.0
        gamma_corrected = np.power(img_array / 255.0, 1.0 / gamma) * 255.0
        gamma_corrected = gamma_corrected.astype(np.uint8)
        variations.append(self._prepare_tensor(Image.fromarray(gamma_corrected)))
        
        return variations

    def _get_occlusion_variations(self, img):
        """Get optimized variations for occlusion handling"""
        variations = []
        img_array = np.array(img)
        
        # Enhanced shadow removal using aggressive CLAHE
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Single optimized CLAHE for occlusions
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        shadow_removed = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        variations.append(self._prepare_tensor(Image.fromarray(shadow_removed)))
        
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
            # Pre-allocate lists for better memory efficiency
            gpu_tensors = []
            gpu_tensors.extend(
                (self.preprocess_image(img), bbox)
                for img, bbox in image_bbox_pairs
                if img is not None
            )

            # Process in smaller sub-batches with overlap
            total_sub_batches = (len(gpu_tensors) + queue_size - 1) // queue_size
            print(f"Processing {len(gpu_tensors)} images in {total_sub_batches} sub-batches...")

            # Process batches with slight overlap for better GPU utilization
            overlap = min(32, queue_size // 4)  # 25% overlap or 32, whichever is smaller
            for i in range(0, len(gpu_tensors), queue_size - overlap):
                sub_batch = gpu_tensors[i:i + queue_size]
                batch_detections = self._process_tensors(sub_batch)
                all_detections.extend(batch_detections)
                
                # Explicit cleanup
                torch.cuda.synchronize()  # Ensure GPU operations are complete
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
            # Pre-allocate lists for efficiency
            for tensor_variations, bbox in tensor_batch:
                batch_boxes = []
                
                # Process each variation
                for i, tensor in enumerate(tensor_variations):
                    # Direct numpy conversion without unnecessary transfers
                    input_tensor = tensor.unsqueeze(0).cpu().numpy()
                    outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
                    boxes = outputs[0][0]
                    
                    # Quick confidence filtering
                    conf_adjustment = self._get_confidence_adjustment(i, len(tensor_variations))
                    boxes[:, 4] *= conf_adjustment
                    conf_mask = boxes[:, 4] > self.confidence_threshold
                    
                    if conf_mask.any():
                        batch_boxes.append(boxes[conf_mask])
                
                # Process combined detections if any exist
                if batch_boxes:
                    combined_boxes = np.concatenate(batch_boxes, axis=0)
                    boxes_tensor = torch.from_numpy(combined_boxes).cuda()
                    centers = boxes_tensor[:, :2] / 640
                    
                    # Calculate coordinates efficiently
                    lon_offset = bbox[2] - bbox[0]
                    lat_offset = bbox[3] - bbox[1]
                    lons = bbox[0] + (centers[:, 0] * lon_offset)
                    lats = bbox[3] - (centers[:, 1] * lat_offset)
                    confs = boxes_tensor[:, 4]
                    
                    # Batch process coordinates
                    coords = torch.stack([lons, lats, confs], dim=1).cpu().numpy()
                    detections.extend([
                        {'lon': lon, 'lat': lat, 'confidence': float(conf)}
                        for lon, lat, conf in coords
                    ])
                    
                    del boxes_tensor, coords
                
                del batch_boxes
                
            return detections
            
        except Exception as e:
            print(f"Error processing tensor batch: {str(e)}")
            return []
        finally:
            torch.cuda.empty_cache()

    def _get_confidence_adjustment(self, variation_index, total_variations):
        """Optimized confidence adjustments for shadow detection"""
        # Confidence weights for each variation
        adjustments = {
            0: 1.0,    # Original image
            1: 0.95,   # CLAHE shadow enhancement
            2: 0.90,   # High brightness
            3: 0.92,   # Gamma correction
            4: 0.88    # Occlusion CLAHE
        }
        return adjustments.get(variation_index, 0.85)

    def cleanup(self):
        """Clean up GPU resources"""
        torch.cuda.empty_cache()
        gc.collect()