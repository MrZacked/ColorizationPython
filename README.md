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

colorization_release_v2.caffemodel: https://www.dropbox.com/scl/fi/d8zffur3wmd4wet58dp9x/colorization_release_v2.caffemodel?rlkey=iippu6vtsrox3pxkeohcuh4oy&dl=0