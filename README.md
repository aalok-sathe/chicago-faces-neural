## A repository to use the Chicago Face Database to train neural networks.
[![Keras](https://img.shields.io/badge/framework-keras-red.svg)](https://keras.io)
[![Python ver](https://img.shields.io/pypi/pyversions/Django.svg)](https://www.python.org/)
[//]: # "[![OpenCV2](https://img.shields.io/badge/uses-opencv2,%20numpy,%20progressbar-fb1ffb.svg)](https://pypi.org/)"


Note, CFD is not provided here in its original form due to copyright reasons.
You will need to obtain your own copy from
[here](http://faculty.chicagobooth.edu/bernd.wittenbrink/cfd/index.html) and
organize and rename some of the files in the manner described below.
However, pickled data is kept in repo as git-lfs pointers, and contains implicit
pre-processed image matrices which can be used on-the-go for training.

### Requisite

* clone repo into a directory, say, `chicago-faces-neural`. 
* install dependencies (or simply run program to find out what is missing
    through runtime exceptions!)
    * [`tensorflow`](https://www.tensorflow.org/)
    * [`keras`](https://keras.io/)
    * [`numpy`](http://www.numpy.org/)
    * [Matplotlib::`pyplot`](https://matplotlib.org/api/pyplot_api.html)
    * [`progressbar`](https://pypi.org/project/progressbar2/)
    * [`opencv2`](https://pypi.org/project/opencv-python/)

### Usage

* To train an ACGAN on genders (currently race and emotion are not implemented):
    * `python3 faces_acgan.py`
        * std output: displays training progress and performance
        * disk output: intermittently saves `x.jpg` to `code/images/<timestamp>/`
          where `x` is the epoch number. sampling frequency may be adjusted
          using the source file. saves model description as
          `saved_model/<type><timestamp>.json` and weights as
          `saved_model/<type><timestamp>.h5` where
          `<type>` may be *generator* or *discriminator*.
* To train a discriminator on race, gender, and emotion:
    * `python3 face_labeler.py`
        * std output: displays training progress and model evaluation on test
          data
        * disk output: saves model once finished training as 
          `saved_model/labeler<timestamp>.{json,h5}`.

### Examples

#### Discriminator

What follows is the output from one particular run of the discriminator.
The model was trained on 80% of all data in 100 epochs with a batch size of 16.
The model was evaluated on the remaining 20% of the data and produced
the results that follow. [As of commit 2b5bac].

| net loss            | `race' loss         | `gender' loss        | `emotion' loss      | `race' accuracy     | `gender' accuracy   |  `emotion' accuracy |
|---------------------|---------------------|----------------------|---------------------|---------------------|---------------------|---------------------|
| 1.1508 | 0.2596 | 0.0855 | 0.8055 | 0.8925 | 0.9669 | 0.7685  |

### Notes

The files `faces_acgan.py` and `face_labeler.py` are the primary files
with an implementation of (1) an auxiliary classifier generative adversarial
network (AC-GAN) and a (2) convolutional discrimator network. The file
`get_face.py` is intended for indexing all the images of the CFD, and norming
them to the input required by the ACGAN. The same module provides a method to
easily import data into another script, similar to the `keras.datasets.mnist`
module's `load_data` method. `get_face.py` supports API calls to supply images
of custom dimensions, and grayscale or BGR. The module supplies image matrices
along with their labels such that they may be reshaped and fed to a network for
training. The module also supports generating lists of references to images,
with none to all of race, gender, and emotion specified. When a particular
argument is left unspecified, the module will list all of its combinations.

Although original images are not supplied, processed, and highly reduced
images are stored as pickled dictionaries, in a way suitable to supply to the
training script: in grayscale (single channel) and reduced to 100x100.

### Repo structure (output of `tree [options]`)

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

The directory `data/images/` contains the raw images, obtained as-are, with the same
names. The file `data.csv` is a csv export of the *first* page of the `.xlsx`
provided in the CFD. You *will* need the `.csv` file for parts of this program,
but you may skip generating one for now.
The `CFD 2.0.3 Norming Data and Codebook.ods` file is simply an Open Document
Format conversion of the original `.xlsx`, and is not necessary for the program.