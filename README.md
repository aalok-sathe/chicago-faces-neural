## A repository to use the Chicago Face Database to train neural networks.

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

Note: `.../images` contains the raw images, obtained as-are, with the same
names. The file `data.csv` is a csv export of the *first* page of the `.xlsx`
provided in the CFD. The `CFD 2.0.3 Norming Data and Codebook.ods` file is
simply an open document format conversion, and is optional.