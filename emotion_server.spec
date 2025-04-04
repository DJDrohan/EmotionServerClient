from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import torch
import cv2

import PyInstaller.config
PyInstaller.config.CONF['excludedimports'] = ['tensorflow', 'tensorflow.keras']

# Collect necessary submodules from libraries
hiddenimports = collect_submodules('torch') + collect_submodules('flask') + collect_submodules('cv2') + [
    'model',        # Custom script for the CNN model
    'resize_image', # Custom image resizing utility
    'hash_utils',   # Custom hashing utilities
    'EmotionLabel', # Custom label drawing utilities
]

a = Analysis(
    ['serverjson.py'],  # Main script
    pathex=[],  # Additional paths if needed
    binaries=[],  # Any extra binaries
    datas=[  # Include necessary non-Python files
        ('models/67e 76p/best_emotion_cnn.pth', 'models/67e 76p/'),  # Model weights file
        (torch.hub.get_dir(), 'torch/hub/'),  # Torch cache directory
        (cv2.data.haarcascades + 'haarcascade_frontalface_default.xml', 'cv2/data/'),  # Haar cascade file
        ("templates/server.html", "templates/"),  # HTML template
    ],
    hiddenimports=hiddenimports,  # Include hidden imports
    hookspath=[],
    runtime_hooks=[], 
    excludes=['tensorflow', 'tensorflow.keras'],
    noarchive=False  # Disable archive mode
)

# Create the executable
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='Emotion_Server',
    debug=False,
    strip=True,
    upx=True,
    console=True
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    name='Emotion_Server'
)
