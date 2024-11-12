"""Central configuration for the car detection system"""

DEFAULT_CONFIG = {
    # WMS settings
    'wms_url': "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0",
    'wms_layer': 'Actueel_orthoHR',
    'wms_srs': 'EPSG:4326',
    'wms_size': (640, 640),  # Size to fetch from WMS
    'model_input_size': (640, 640),  # Size for model processing
    'wms_format': 'image/jpeg',
    
    # Processing settings
    'tile_size_meters': 64.0,
    'confidence_threshold': 0.3,
    'tile_overlap': 0.1,
    'batch_size': 64,
    'checkpoint_interval': 2000,
    'max_gpu_memory': 2.0,
    'duplicate_distance': 1.0,
    'num_workers': 25,
    'queue_size': 64,
    
    # Default paths
    'frame_path': 'amsterdam.shp',
    'model_path': 'car_aerial_detection_yolo7_ITCVD_deepness.onnx',
    
    # Output settings
    'output_prefix': 'detections'
} 