## A repository to use the Chicago Face Database to train neural networks.
[![Keras](https://img.shields.io/badge/framework-keras-red.svg)](https://keras.io)
[![Python ver](https://img.shields.io/pypi/pyversions/Django.svg)](https://www.python.org/)
[![OpenCV2](https://img.shields.io/badge/uses-opencv2,%20numpy,%20progressbar-fb1ffb.svg)](https://pypi.org/)


Note, CFD is not provided here due to copyright reasons.
You will need to obtain your own copy from
[here](http://faculty.chicagobooth.edu/bernd.wittenbrink/cfd/index.html) and
organize and rename some of the files in the manner described below.

The file `faces_acgan.py` is the primary file with an implementation of an
auxiliary classifier generative adversarial network (AC-GAN). The file
`get_face.py` is intended for indexing all the images of the CFD, and norming
them to the input required by the ACGAN. The same module provides a method to
easily import data into another script, similar to the `keras.datasets.mnist`
module's `load_data` method.

Although original images are not supplied, processed, and highly reduced
images are stored as pickled dictionaries, in a way suitable to supply to the
training script: in grayscale (single channel) and reduced to 100x100.

```bash
.
├── code
│   ├── cp_imgs.sh
│   ├── face_labeler.py
│   ├── faces_acgan.py
│   ├── get_face.py
│   ├── images [not opening dir]
│   ├── pickled
│   │   ├── image_containers.pickle
│   │   ├── images.pickle
│   │   └── indexed_faces.pickle
│   ├── runtime.kerasconfig
│   └── saved_model [not opening dir]
├── data
│   └── cfd2.0.3
│       ├── CFD 2.0.3 Norming Data and Codebook.ods
│       ├── CFD 2.0.3 Read Me.pdf
│       ├── data.csv
│       └── images [597 entries exceeds filelimit, not opening dir]
├── LICENSE
├── README.md
└── readme.pdf
```

Note: `<path>/images/` contains the raw images, obtained as-are, with the same
names. The file `data.csv` is a csv export of the *first* page of the `.xlsx`
provided in the CFD. You *will* need the `.csv` file for parts of this program,
but you may skip generating one for now.
The `CFD 2.0.3 Norming Data and Codebook.ods` file is simply an Open Document
Format conversion of the original `.xlsx`, and is not necessary for the program.