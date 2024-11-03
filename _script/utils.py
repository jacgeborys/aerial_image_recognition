import os
import json
import geopandas as gpd
from shapely.geometry import Point
import math
from pyproj import Transformer
from shapely.ops import transform
import time

class TileGenerator:
    @staticmethod
    def generate_tiles(bounds, tile_size_meters=64.0, overlap=0.1):
        """Generate tile coordinates with overlap"""
        minx, miny, maxx, maxy = bounds
        mid_lat = (miny + maxy) / 2

        # Convert meters to degrees
        earth_radius = 6378137
        lat_deg = tile_size_meters / (earth_radius * math.pi / 180)
        lon_deg = tile_size_meters / (earth_radius * math.pi / 180 * math.cos(math.radians(mid_lat)))

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


class CheckpointManager:
    def __init__(self, checkpoint_dir, prefix=''):
        self.checkpoint_dir = checkpoint_dir
        self.prefix = f"{prefix}_" if prefix else ""
        self.state_file = os.path.join(checkpoint_dir, f"{self.prefix}processing_state.json")
        self.data_file = os.path.join(checkpoint_dir, f"{self.prefix}latest_detections.geojson")
        self.temp_state_file = os.path.join(checkpoint_dir, f"{self.prefix}temp_state.json")
        self.temp_data_file = os.path.join(checkpoint_dir, f"{self.prefix}temp_detections.geojson")

    def load_checkpoint(self):
        """Load last processing state and detections"""
        last_index = 0
        previous_detections = []

        if os.path.exists(self.state_file) and os.path.exists(self.data_file):
            try:
                # Load state
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                last_index = state.get('last_processed_index', 0)

                # Load previous detections
                if os.path.exists(self.data_file):
                    gdf = gpd.read_file(self.data_file)
                    if not gdf.empty:
                        previous_detections = [
                            {
                                'lon': point.x,
                                'lat': point.y,
                                'confidence': conf
                            }
                            for point, conf in zip(gdf.geometry, gdf.confidence)
                        ]
                        print(f"\nLoaded {len(previous_detections)} previous detections")

            except Exception as e:
                print(f"\nCheckpoint load error: {str(e)}")
                return 0, []

        return last_index, previous_detections

    def save_checkpoint(self, detections, processed_tiles, total_tiles):
        """Save processing state and detections atomically"""
        try:
            # Save to temporary files
            if detections:
                gdf = self._create_geodataframe(detections)
                gdf.to_file(self.temp_data_file, driver='GeoJSON')

            state = {
                'processed_tiles': processed_tiles,
                'total_tiles': total_tiles,
                'last_processed_index': processed_tiles - 1,
                'timestamp': time.time(),
                'num_detections': len(detections)
            }

            with open(self.temp_state_file, 'w') as f:
                json.dump(state, f)

            # Move temporary files to final location
            if os.path.exists(self.temp_data_file):
                if os.path.exists(self.data_file):
                    os.remove(self.data_file)
                os.rename(self.temp_data_file, self.data_file)

            if os.path.exists(self.temp_state_file):
                if os.path.exists(self.state_file):
                    os.remove(self.state_file)
                os.rename(self.temp_state_file, self.state_file)

            print(f"\nCheckpoint saved: {processed_tiles}/{total_tiles} tiles, {len(detections)} detections")
            return True

        except Exception as e:
            print(f"Checkpoint save error: {str(e)}")
            # Cleanup temporary files
            for temp_file in [self.temp_state_file, self.temp_data_file]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            return False

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
class ResultsManager:
    def __init__(self, duplicate_distance=2.0):
        self.duplicate_distance = duplicate_distance

    def remove_duplicates(self, detections):
        """Remove duplicate detections"""
        if not detections:
            return []

        try:
            gdf = self._create_geodataframe(detections)

            # Convert to UTM for distance calculations
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
            gdf['geometry'] = gdf['geometry'].apply(
                lambda p: transform(transformer.transform, p)
            )

            # Sort by confidence
            gdf = gdf.sort_values('confidence', ascending=False)
            kept_indices = self._find_unique_points(gdf)

            # Convert back to WGS84
            transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
            filtered_gdf = gdf.loc[kept_indices]
            filtered_gdf['geometry'] = filtered_gdf['geometry'].apply(
                lambda p: transform(transformer.transform, p)
            )

            return filtered_gdf.to_dict('records')

        except Exception as e:
            print(f"Deduplication error: {str(e)}")
            return detections

    def _find_unique_points(self, gdf):
        """Find unique points based on distance threshold"""
        kept_indices = []
        for idx in gdf.index:
            if idx in kept_indices:
                continue

            point = gdf.loc[idx, 'geometry']
            distances = gdf['geometry'].apply(lambda p: point.distance(p))
            duplicates = distances[distances < self.duplicate_distance].index
            kept_indices.extend(duplicates)

        return kept_indices

    def _create_geodataframe(self, detections):
        """Convert detections to GeoDataFrame"""
        points = [Point(d['lon'], d['lat']) for d in detections]
        confs = [d['confidence'] for d in detections]
        return gpd.GeoDataFrame(
            {'geometry': points, 'confidence': confs},
            crs="EPSG:4326"
        )