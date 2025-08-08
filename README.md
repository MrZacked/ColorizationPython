Colorization
============

Batch colorize grayscale photos in a folder.

Setup
-----
1) Python 3.10+
2) Install deps:
```
pip install -r requirements.txt
```

Models
------
Put your Caffe model file in `models/`, for example:
`models/colorization_release_v2.caffemodel`

The script auto-downloads the prototxt and `pts_in_hull.npy`.

Run
---
Defaults:
- input: `NonColorImg/`
- output: `ColorizedOut/`

Examples:
```
python colorize.py --caffemodel models/colorization_release_v2.caffemodel
python colorize.py --input_dir NonColorImg --output_dir ColorizedOut --caffemodel models/colorization_release_v2.caffemodel
```

Notes
-----
- Works on CPU.
- You must supply the `.caffemodel` yourself.
