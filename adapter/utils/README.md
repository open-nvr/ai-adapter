# adapter/utils/ -- Shared Utilities

Helper functions used across model handlers.

## Files

```
utils/
├── image_utils.py      # Load images from opennvr:// URIs
├── visualization.py    # Draw bounding boxes on images
└── __init__.py
```

## image_utils.py

### `load_image_from_uri(uri: str) -> np.ndarray`

Converts an `opennvr://` URI to a file path, loads the image with OpenCV, and returns it as a numpy array (BGR format).

**URI format:** `opennvr://frames/<camera_id>/<filename>`
**Example:** `opennvr://frames/camera_0/latest.jpg`

**Security:** Validates path containment to prevent directory traversal attacks. The resolved path must stay inside `BASE_FRAMES_DIR`.

**Errors:**
- `400` if URI format is invalid or path traversal is detected
- `404` if the frame file doesn't exist
- `400` if the image can't be decoded

### `validate_image(img) -> bool`

Simple check that an image loaded correctly (not None).

## visualization.py

### `draw_bounding_boxes(image, detections, count, show_labels, show_count) -> np.ndarray`

Draws colorful bounding boxes on an image for visual debugging and annotated output.

**Parameters:**
- `image` -- source image (BGR numpy array)
- `detections` -- list of dicts, each with `bbox` ([left, top, width, height]) and `confidence`
- `count` -- total count to display in corner (optional)
- `show_labels` -- whether to label each box with "Person N: 0.85"
- `show_count` -- whether to show "Count: N" in top-left

Uses an 8-color palette that cycles for multiple detections. Returns a copy (doesn't modify the original).
