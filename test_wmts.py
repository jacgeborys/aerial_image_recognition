from owslib.wmts import WebMapTileService
import requests
import os
from pyproj import Transformer
from datetime import datetime

def calculate_tile_indices(x, y, matrix):
    """Calculate tile indices using matrix bounds"""
    resolution = matrix.scaledenominator * 0.00028
    tile_width = resolution * matrix.tilewidth
    tile_height = resolution * matrix.tileheight
    
    # Matrix bounds
    bounds_left = 100000.0
    bounds_top = 850000.0
    
    # Calculate position relative to bounds
    dx = x - bounds_left
    dy = bounds_top - y
    
    # Calculate indices
    col = int(dx / tile_width)
    row = int(dy / tile_height)
    
    print(f"\nCalculating indices:")
    print(f"  Target point: ({x}, {y})")
    print(f"  Matrix bounds: left={bounds_left}, top={bounds_top}")
    print(f"  Resolution: {resolution:.2f}m/px")
    print(f"  Tile size: {tile_width:.2f}m x {tile_height:.2f}m")
    print(f"  Distance from bounds: dx={dx:.1f}m, dy={dy:.1f}m")
    print(f"  Raw indices: col={dx/tile_width:.2f}, row={dy/tile_height:.2f}")
    
    return col, row

def try_tile_range(wmts, zoom_level, center_col, center_row, radius=2):
    """Try a range of tiles around a center point"""
    matrix = wmts.tilematrixsets['EPSG:2180'].tilematrix[zoom_level]
    resolution = matrix.scaledenominator * 0.00028
    tile_width = resolution * matrix.tilewidth
    tile_height = resolution * matrix.tileheight
    
    print(f"\nSearching tiles at zoom {zoom_level}")
    print(f"Matrix size: {matrix.matrixwidth}x{matrix.matrixheight}")
    print(f"Resolution: {resolution:.2f}m/px")
    print(f"Tile size: {tile_width:.2f}m x {tile_height:.2f}m")
    print(f"Center point: row={center_row}, col={center_col}")
    
    os.makedirs("test_tiles", exist_ok=True)
    
    bounds_left = 100000.0
    bounds_top = 850000.0
    
    for delta_row in range(-radius, radius + 1):
        for delta_col in range(-radius, radius + 1):
            row = center_row + delta_row
            col = center_col + delta_col
            
            if row < 0 or col < 0 or row >= matrix.matrixheight or col >= matrix.matrixwidth:
                continue
            
            # Calculate tile coordinates based on matrix bounds
            tile_x = bounds_left + (col * tile_width + tile_width/2)
            tile_y = bounds_top - (row * tile_height + tile_height/2)
            
            print(f"\nTile at row={row}, col={col}")
            print(f"  Center EPSG:2180: ({tile_x:.1f}, {tile_y:.1f})")
            
            # Convert to WGS84 for verification
            transformer = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(tile_x, tile_y)
            print(f"  Center WGS84: {lon:.6f}°, {lat:.6f}°")
            
            url = (
                f"{wmts.url}?service=WMTS"
                f"&request=GetTile"
                f"&version=1.0.0"
                f"&layer=ORTOFOTOMAPA"
                f"&style=default"
                f"&format=image/jpeg"
                f"&tileMatrixSet=EPSG:2180"
                f"&tileMatrix={zoom_level}"
                f"&tileRow={row}"
                f"&tileCol={col}"
            )
            
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 QGIS/3.22.16'})
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                content_length = len(response.content)
                
                if content_length > 1000 and 'image' in content_type:
                    zoom_num = zoom_level.split(':')[-1]
                    filename = f"test_tiles/warsaw_z{zoom_num}_r{row}_c{col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print(f"  ✓ Saved tile ({content_length} bytes)")
                else:
                    print(f"  × Invalid response: {content_type}, {content_length} bytes")
            else:
                print(f"  × Failed: HTTP {response.status_code}")

def main():
    url = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMTS/StandardResolution"
    wmts = WebMapTileService(url)
    
    # Target point
    target_x, target_y = 635332.5, 491203.0
    print(f"Target point: ({target_x}, {target_y})")
    
    # Try both zoom levels
    for zoom_level in ['EPSG:2180:7', 'EPSG:2180:14']:
        matrix = wmts.tilematrixsets['EPSG:2180'].tilematrix[zoom_level]
        col, row = calculate_tile_indices(target_x, target_y, matrix)
        print(f"Zoom {zoom_level} indices: col={col}, row={row}")
        try_tile_range(wmts, zoom_level, col, row, radius=1)

if __name__ == "__main__":
    main()