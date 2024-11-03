import os
import geopandas as gpd
from shapely.geometry import box, Point

class CoordinateDebugger:
    def __init__(self, output_dir='.'):
        self.tiles_file = os.path.join(output_dir, 'tile_debug.geojson')
        self.detections_file = os.path.join(output_dir, 'detection_debug.geojson')
        self.log_file = os.path.join(output_dir, 'coordinate_debug.csv')
        
        # Initialize files
        self._init_csv()
        self._init_geojson()
        
    def _init_csv(self):
        with open(self.log_file, 'w') as f:
            f.write("step,tile_id,tile_type,minx,miny,maxx,maxy,centroid_x,centroid_y\n")
            
    def _init_geojson(self):
        # Create empty GeoJSON files for tiles and detections
        empty_gdf = gpd.GeoDataFrame(columns=['geometry', 'tile_id', 'tile_type'], crs="EPSG:4326")
        empty_gdf.to_file(self.tiles_file, driver='GeoJSON')
        empty_gdf.to_file(self.detections_file, driver='GeoJSON')
    
    def log_tile(self, step, tile_id, tile_type, bbox):
        # Log to CSV
        centroid_x = (bbox[0] + bbox[2]) / 2
        centroid_y = (bbox[1] + bbox[3]) / 2
        with open(self.log_file, 'a') as f:
            f.write(f"{step},{tile_id},{tile_type},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{centroid_x},{centroid_y}\n")
        
        # Add to GeoJSON
        polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])
        gdf = gpd.GeoDataFrame({
            'geometry': [polygon],
            'tile_id': [tile_id],
            'tile_type': [tile_type]
        }, crs="EPSG:4326")
        gdf.to_file(self.tiles_file, driver='GeoJSON', mode='a')
    
    def log_detection(self, detection_id, bbox, final_coords):
        # Create point for detection
        point = Point(final_coords[0], final_coords[1])
        # Create polygon for source tile
        tile = box(bbox[0], bbox[1], bbox[2], bbox[3])
        
        gdf = gpd.GeoDataFrame({
            'geometry': [point, tile],
            'detection_id': [detection_id, detection_id],
            'type': ['detection', 'source_tile']
        }, crs="EPSG:4326")
        gdf.to_file(self.detections_file, driver='GeoJSON', mode='a') 