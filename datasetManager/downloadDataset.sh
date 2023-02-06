#!/bin/bash
# COCO 2017 dataset http://cocodataset.org
# Download command: bash ./scripts/get_coco.sh

# Download/unzip labels
d='./' # unzip directory
url=https://cnrsc-my.sharepoint.com/:f:/g/personal/marco_delcoco_cnr_it/ErrKmxDVvvBDrMIz721Cj_sBRdyoWi8g-UErVGszL6iY-w?e=QKSU7O
f='coco2017labels-segments.zip' # or 'coco2017labels.zip', 68 MB
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background

wait # finish background tasks