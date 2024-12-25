# KGOT-image-augmentation

Prelimary file structure, ultimate goal: allow user to try kgot from our app.

project_root/
├── kgot/
│   ├── __init__.py
│   ├── utils.py                # General utility functions (interpolation, tangents, etc.)
│   ├── coco_processing.py      # Functions to handle COCO JSON files and masks
│   ├── straighten.py           # Functions for straightening images and processing masks
│   ├── visualization.py        # Functions for image and curve visualization
│   └── application.py          # Application logic (upload image, run process, save results)
├── app.py                      # Main script for running the application
├── config.py                   # Configuration file for parameters (e.g., paths, defaults)
└── requirements.txt            # List of dependencies
