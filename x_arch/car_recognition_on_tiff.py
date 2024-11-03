from samgeo import SamGeo2
import numpy
import os

# Set the path to your image
image_path = r"C:\Users\Asus\OneDrive\Pulpit\Rozne\QGIS\car_recognition\img\praga.tif"

# Set the output directory
output_dir = os.path.dirname(image_path)

# Initialize SamGeo2
sam2 = SamGeo2(
    model_id="sam2-hiera-large",
    apply_postprocessing=False,
    points_per_side=32,
    points_per_batch=64,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25.0,
    use_m2m=True,
)

# Generate masks
print("Generating masks...")
masks = sam2.generate(image_path)

# Save masks as a GeoTIFF
print("Saving masks as GeoTIFF...")
geotiff_output = os.path.join(output_dir, "output_masks.tif")
sam2.save_masks(image_path, masks, output=geotiff_output)

# Save masks as vectors (polygons)
print("Saving masks as vectors...")
vector_output = os.path.join(output_dir, "output_vectors.gpkg")
sam2.masks_to_vectors(image_path, masks, output=vector_output)

print("Processing complete!")
print(f"GeoTIFF saved to: {geotiff_output}")
print(f"Vector file saved to: {vector_output}")

# Optionally, visualize the masks
sam2.show_masks(image_path, masks)