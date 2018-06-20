#!/bin/bash

### create a gzipped tarball of contents of images at host location
ssh vaio /bin/bash << EOF
    cd ~/projects/chicago-faces-neural/code/images
    tar czfv images_contents.tar.gz *
EOF

### back to home location
cd ~/code/chicago-faces/code/images/
### mark everything R/O
#chmod a-w ~/code/chicago-faces/code/images/*

### remove existing contents
rm -rf ~/code/chicago-faces/code/images/*

### copy over tarball and unzip, and get rid of tar
scp vaio:~/projects/chicago-faces-neural/code/images/images_contents.tar.gz .
tar xvzf images_contents.tar.gz
rm -f images_contents.tar.gz

### delete remote copy of tarball
ssh vaio /bin/bash << EOF
    rm -f ~/projects/chicago-faces-neural/code/images/images_contents.tar.gz
EOF

