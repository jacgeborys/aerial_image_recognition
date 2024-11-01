# Import main components for easy access
from .detector import CarDetector
from .monitors import GPUMonitor
from .wms_handler import WMSHandler
from .gpu_handler import GPUHandler
from .utils import TileGenerator, CheckpointManager, ResultsManager

# Define what should be available when someone does 'from _script import *'
__all__ = [
    'CarDetector',
    'GPUMonitor',
    'WMSHandler',
    'GPUHandler',
    'TileGenerator',
    'CheckpointManager',
    'ResultsManager'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'Your Name'