## A repository to use the Chicago Face Database to train neural networks.
[![Keras](https://img.shields.io/badge/framework-keras-red.svg)](https://keras.io)
[![Python ver](https://img.shields.io/pypi/pyversions/Django.svg)](https://www.python.org/)


Note, CFD is not provided here due to copyright reasons.
You will need to obtain your own copy from
[here](http://faculty.chicagobooth.edu/bernd.wittenbrink/cfd/index.html) and
organize and rename some of the files in the manner described below.

```bash
.
├── code
│   └── faces-acgan.py
└── database
    └── cfd2.0.3
        ├── CFD 2.0.3 Norming Data and Codebook.ods
        ├── CFD 2.0.3 Read Me.pdf
        ├── data.csv
        └── images [597 entries]
```

Note: `<path>/images/` contains the raw images, obtained as-are, with the same
names. The file `data.csv` is a csv export of the *first* page of the `.xlsx`
provided in the CFD. You *will* need the `.csv` file for this program.
The `CFD 2.0.3 Norming Data and Codebook.ods` file is simply an Open Document
Format conversion of the original `.xlsx`, and is not necessary for the program.