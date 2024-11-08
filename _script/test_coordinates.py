from pyproj import Transformer

def test_coordinates():
    """Test coordinate transformations with sample values"""
    # Sample point from your bbox (center of first tile)
    x_2180 = 636041.66  # (minx + maxx) / 2
    y_2180 = 491551.42  # (miny + maxy) / 2
    
    # Create transformers
    to_4326 = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    direct_to_3857 = Transformer.from_crs("EPSG:2180", "EPSG:3857", always_xy=True)
    
    # Test transformations
    # Method 1: Through WGS84
    lon, lat = to_4326.transform(x_2180, y_2180)
    x_3857, y_3857 = to_3857.transform(lon, lat)
    print(f"\nMethod 1 (through WGS84):")
    print(f"EPSG:2180: {x_2180:.2f}, {y_2180:.2f}")
    print(f"EPSG:4326: {lon:.6f}, {lat:.6f}")
    print(f"EPSG:3857: {x_3857:.2f}, {y_3857:.2f}")
    
    # Method 2: Direct to Web Mercator
    x_3857_direct, y_3857_direct = direct_to_3857.transform(x_2180, y_2180)
    print(f"\nMethod 2 (direct to Web Mercator):")
    print(f"EPSG:3857: {x_3857_direct:.2f}, {y_3857_direct:.2f}")
    
    # Test full bbox transformation
    bbox_2180 = (636009.66, 491519.42, 636073.66, 491583.42)
    minx_3857, miny_3857 = direct_to_3857.transform(bbox_2180[0], bbox_2180[1])
    maxx_3857, maxy_3857 = direct_to_3857.transform(bbox_2180[2], bbox_2180[3])
    
    print(f"\nFull bbox in EPSG:3857:")
    print(f"minx: {minx_3857:.2f}, miny: {miny_3857:.2f}")
    print(f"maxx: {maxx_3857:.2f}, maxy: {maxy_3857:.2f}")
    print(f"width: {maxx_3857 - minx_3857:.2f}m")
    print(f"height: {maxy_3857 - miny_3857:.2f}m")

test_coordinates()