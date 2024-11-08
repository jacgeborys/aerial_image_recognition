import os
import json
import geopandas as gpd
from shapely.geometry import Point
import math
from pyproj import Transformer
from shapely.ops import transform
import time
import numpy as np
from datetime import datetime
import pickle

__all__ = ['TileGenerator', 'CheckpointManager', 'ResultsManager', 'create_geodataframe']

class TileGenerator:
    @staticmethod
    def _get_utm_zone(lon, lat):
        """Calculate UTM zone for given coordinates"""
        utm_zone = int((lon + 180) / 6) + 1
        return f"326{utm_zone}" if lat >= 0 else f"327{utm_zone}"

    @staticmethod
    def generate_tiles(bounds, tile_size_meters=64.0, overlap=0.1):
        """Generate tile coordinates with UTM-based accuracy"""
        minx, miny, maxx, maxy = bounds
        mid_lon = (minx + maxx) / 2
        mid_lat = (miny + maxy) / 2
        
        # Get UTM zone for the area
        utm_epsg = TileGenerator._get_utm_zone(mid_lon, mid_lat)
        
        # Create transformers
        from_wgs84 = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
        to_wgs84 = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
        
        # Convert bounds to UTM
        utm_minx, utm_miny = from_wgs84.transform(minx, miny)
        utm_maxx, utm_maxy = from_wgs84.transform(maxx, maxy)
        
        # Calculate steps in meters
        step_size = tile_size_meters * (1 - overlap)
        
        tiles = []
        utm_x = utm_minx
        while utm_x < utm_maxx:
            utm_y = utm_miny
            while utm_y < utm_maxy:
                # Calculate tile bounds in UTM
                tile_utm_bounds = (
                    utm_x,
                    utm_y,
                    min(utm_x + tile_size_meters, utm_maxx),
                    min(utm_y + tile_size_meters, utm_maxy)
                )
                
                # Convert back to WGS84
                wgs_x1, wgs_y1 = to_wgs84.transform(tile_utm_bounds[0], tile_utm_bounds[1])
                wgs_x2, wgs_y2 = to_wgs84.transform(tile_utm_bounds[2], tile_utm_bounds[3])
                
                tiles.append((wgs_x1, wgs_y1, wgs_x2, wgs_y2))
                utm_y += step_size
            utm_x += step_size
        
        return tiles


class CheckpointManager:
    def __init__(self, checkpoint_dir, prefix=''):
        self.checkpoint_dir = checkpoint_dir
        self.prefix = f"{prefix}_" if prefix else ""
        self.state_file = os.path.join(checkpoint_dir, f"{self.prefix}processing_state.json")
        self.data_file = os.path.join(checkpoint_dir, f"{self.prefix}latest_detections.geojson")
        self.temp_state_file = os.path.join(checkpoint_dir, f"{self.prefix}temp_state.json")
        self.temp_data_file = os.path.join(checkpoint_dir, f"{self.prefix}temp_detections.geojson")

    def save_checkpoint(self, processed_count, detections, total_tiles):
        """Save checkpoint in readable formats"""
        try:
            # Save processing state as JSON
            state_data = {
                'processed_count': processed_count,
                'total_tiles': total_tiles,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            # Save detections as GeoJSON if there are any
            if detections:
                gdf = self._create_geodataframe(detections)
                gdf.to_file(self.data_file, driver='GeoJSON')

        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")

    def load_checkpoint(self):
        """Load checkpoint from JSON and GeoJSON"""
        try:
            processed_count = 0
            detections = []

            # Load processing state
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                processed_count = state['processed_count']

            # Load detections
            if os.path.exists(self.data_file):
                gdf = gpd.read_file(self.data_file)
                detections = [
                    {
                        'lon': row.geometry.x,
                        'lat': row.geometry.y,
                        'confidence': row.confidence
                    }
                    for _, row in gdf.iterrows()
                ]

            return processed_count, detections

        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return 0, []

    def _create_geodataframe(self, detections):
        """Convert detections to GeoDataFrame"""
        points = []
        confs = []
        
        for d in detections:
            # Skip if this is an image-bbox pair instead of a detection
            if not isinstance(d, dict) or 'lon' not in d:
                continue
                
            points.append(Point(d['lon'], d['lat']))
            confs.append(d['confidence'])
        
        if not points:  # If no valid detections found
            return gpd.GeoDataFrame(columns=['geometry', 'confidence'], crs="EPSG:4326")
            
        return gpd.GeoDataFrame(
            {'geometry': points, 'confidence': confs},
            crs="EPSG:4326"
        )

def create_geodataframe(detections):
    """Convert detections to GeoDataFrame with robust error handling"""
    points = []
    confs = []
    
    for i, d in enumerate(detections):
        try:
            if isinstance(d, dict):
                if 'geometry' in d and 'confidence' in d:
                    points.append(d['geometry'])
                    confs.append(d['confidence'])
                elif 'lon' in d and 'lat' in d:
                    points.append(Point(d['lon'], d['lat']))
                    confs.append(d.get('confidence', 0.0))
                else:
                    print(f"Warning: Skipping invalid detection format at index {i}: {d}")
                    continue
            else:
                print(f"Warning: Skipping non-dictionary detection at index {i}: {d}")
                continue
            
        except Exception as e:
            print(f"Warning: Error processing detection at index {i}: {str(e)}")
            continue
    
    if not points:
        return gpd.GeoDataFrame(columns=['geometry', 'confidence'], crs="EPSG:4326")
        
    return gpd.GeoDataFrame(
        {'geometry': points, 'confidence': confs},
        crs="EPSG:4326"
    )

class ResultsManager:
    def __init__(self, output_dir, prefix="detections", duplicate_distance=1):
        """Initialize with output path and duplicate distance"""
        self.duplicate_distance = duplicate_distance  # meters
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, f"{prefix}_results.geojson")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def process_results(self, detections):
        """Process final detection results"""
        if not detections:
            print("No detections to process")
            return []
            
        print(f"\n[{datetime.now()}] Processing {len(detections)} detections...")
        
        # Remove duplicates
        unique_detections = self.remove_duplicates(detections)
        
        # Create final GeoDataFrame using the utility function
        final_gdf = create_geodataframe(unique_detections)
        
        # Save results
        if not final_gdf.empty:
            final_gdf.to_file(self.output_file, driver='GeoJSON')
            print(f"\nResults saved to: {self.output_file}")
            
        return unique_detections

    def remove_duplicates(self, detections):
        """Remove duplicate detections with cleaner logging"""
        if not detections:
            return []
        
        start_time = time.time()
        initial_count = len(detections)
        
        # Create GeoDataFrame using utility function
        print(f"[{datetime.now()}] Creating GeoDataFrame...")
        gdf = create_geodataframe(detections)

        # Convert to UTM for distance calculations
        print(f"[{datetime.now()}] Converting to UTM...")
        utm_zone = int((gdf.geometry.x.mean() + 180) / 6) + 1
        utm_epsg = f"326{utm_zone}"
        gdf_utm = gdf.to_crs(utm_epsg)

        # Sort by confidence
        gdf_utm = gdf_utm.sort_values('confidence', ascending=False)

        # Create spatial index
        print(f"[{datetime.now()}] Creating spatial index...")
        spatial_index = gdf_utm.sindex

        # Initialize mask for points to keep
        kept_mask = np.ones(len(gdf_utm), dtype=bool)

        print(f"[{datetime.now()}] Finding duplicates...")
        # Iterate through points in order of confidence
        for idx in range(len(gdf_utm)):
            if not kept_mask[idx]:
                continue
            
            # Get current point
            point = gdf_utm.iloc[idx].geometry
            
            # Find potential neighbors using spatial index
            nearby_candidates = list(spatial_index.query(point.buffer(self.duplicate_distance)))
            
            # Remove points that come later (lower confidence)
            for neighbor_idx in nearby_candidates:
                if neighbor_idx > idx and kept_mask[neighbor_idx]:
                    if point.distance(gdf_utm.iloc[neighbor_idx].geometry) < self.duplicate_distance:
                        kept_mask[neighbor_idx] = False

        # Filter and convert back to WGS84
        print(f"[{datetime.now()}] Converting back to WGS84...")
        filtered_gdf = gdf_utm[kept_mask].to_crs("EPSG:4326")

        final_count = len(filtered_gdf)
        if initial_count != final_count:
            print(f"Duplicates removed: {initial_count-final_count} ({(initial_count-final_count)/initial_count*100:.1f}%)")

        # Convert back to the original detection format
        return [
            {
                'lon': point.x,
                'lat': point.y,
                'confidence': conf
            }
            for point, conf in zip(filtered_gdf.geometry, filtered_gdf.confidence)
        ]