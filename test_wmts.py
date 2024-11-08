from owslib.wmts import WebMapTileService
import requests
import os
from datetime import datetime
import geopandas as gpd
from pyproj import Transformer

def get_available_zooms(wmts):
    """Get available zoom levels and their properties"""
    matrix_set = wmts.tilematrixsets['EPSG:2180']
    zooms = []
    
    print("\nAvailable zoom levels:")
    for matrix_id in sorted(matrix_set.tilematrix.keys()):
        matrix = matrix_set.tilematrix[matrix_id]
        resolution = matrix.scaledenominator * 0.00028
        print(f"  {matrix_id}:")
        print(f"    Resolution: {resolution:.2f}m/px")
        print(f"    Matrix size: {matrix.matrixwidth}x{matrix.matrixheight}")
        zooms.append(matrix_id)
    
    return zooms

def fetch_tiles(wmts, location_name, x_3857, y_3857, zoom_level, radius=1):
    """Fetch tiles around a point"""
    matrix = wmts.tilematrixsets['EPSG:2180'].tilematrix[zoom_level]
    resolution = matrix.scaledenominator * 0.00028
    tile_width = resolution * matrix.tilewidth
    tile_height = resolution * matrix.tileheight
    
    # Transform from 3857 to 2180
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:2180", always_xy=True)
    x_2180, y_2180 = transformer.transform(x_3857, y_3857)
    
    # Calculate indices
    dx = x_2180 - 100000.0
    dy = 850000.0 - y_2180
    col = int(dx / tile_width)
    row = int(dy / tile_height)
    
    print(f"\nProcessing {location_name} at zoom {zoom_level}")
    print(f"Resolution: {resolution:.2f}m/px")
    print(f"Tile size: {tile_width:.2f}m x {tile_height:.2f}m")
    print(f"Matrix size: {matrix.matrixwidth}x{matrix.matrixheight}")
    print(f"Target coordinates (2180): ({x_2180:.1f}, {y_2180:.1f})")
    print(f"Tile indices: col={col}, row={row}")
    
    # Create nested directory structure for tiles
    zoom_num = zoom_level.split(':')[-1]
    tile_dir = os.path.join("tiles", location_name, f"z{zoom_num}")
    os.makedirs(tile_dir, exist_ok=True)
    
    # Download tiles
    tiles_downloaded = 0
    
    for delta_row in range(-radius, radius + 1):
        for delta_col in range(-radius, radius + 1):
            test_row = row + delta_row
            test_col = col + delta_col
            
            # Skip if outside matrix bounds
            if test_row < 0 or test_col < 0 or test_row >= matrix.matrixheight or test_col >= matrix.matrixwidth:
                continue
            
            url = (
                f"{wmts.url}?service=WMTS"
                f"&request=GetTile"
                f"&version=1.0.0"
                f"&layer=ORTOFOTOMAPA"
                f"&style=default"
                f"&format=image/jpeg"
                f"&tileMatrixSet=EPSG:2180"
                f"&tileMatrix={zoom_level}"
                f"&tileRow={test_row}"
                f"&tileCol={test_col}"
            )
            
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 QGIS/3.22.16'})
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                content_length = len(response.content)
                
                if content_length > 1000 and 'image' in content_type:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"r{test_row}_c{test_col}_{timestamp}.jpg"
                    filepath = os.path.join(tile_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    print(f"✓ Tile saved: row={test_row}, col={test_col}")
                    tiles_downloaded += 1
                else:
                    print(f"× Invalid response for row={test_row}, col={test_col}")
            else:
                print(f"× Failed for row={test_row}, col={test_col}: {response.status_code}")
    
    print(f"Downloaded {tiles_downloaded} tiles for zoom level {zoom_level}")
    return tiles_downloaded > 0

def main():
    # WMTS service URL
    url = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMTS/StandardResolution"
    wmts = WebMapTileService(url)
    
    # Get available zoom levels
    available_zooms = get_available_zooms(wmts)
    print("\nFound zoom levels:", available_zooms)
    
    # Input shapefiles (all in EPSG:3857)
    input_dir = r"C:\Users\Asus\OneDrive\Pulpit\Rozne\QGIS\car_recognition\gis\frames\3857"
    locations = {
        'warsaw': 'warsaw_3857.shp',
        'gdansk': 'gdansk_3857.shp',
        'lublin': 'lublin_3857.shp'
    }
    
    # Process each location
    for name, filename in locations.items():
        print(f"\nProcessing {name}")
        
        # Read shapefile
        shapefile_path = os.path.join(input_dir, filename)
        gdf = gpd.read_file(shapefile_path)
        
        if str(gdf.crs) != "EPSG:3857":
            print(f"Warning: {name} is not in EPSG:3857")
            continue
        
        # Get centroid
        centroid = gdf.geometry.union_all().centroid
        
        # Try all zoom levels that exist
        for zoom_level in available_zooms:
            if zoom_level.startswith('EPSG:2180:') and int(zoom_level.split(':')[-1]) >= 14:
                try:
                    # Adjust radius based on zoom level
                    zoom_num = int(zoom_level.split(':')[-1])
                    radius = min(2 + (zoom_num - 14), 4)  # Increase radius with zoom, but cap at 4
                    fetch_tiles(wmts, name, centroid.x, centroid.y, zoom_level, radius=radius)
                except Exception as e:
                    print(f"Error processing {zoom_level}: {str(e)}")

if __name__ == "__main__":
    main()