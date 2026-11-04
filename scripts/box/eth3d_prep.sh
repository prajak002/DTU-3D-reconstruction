#!/bin/bash
# ETH3D: per scene, make sure the depth maps are there, preprocess, then delete the raw scene to save disk.
source /home/ubuntu/env.sh
RAW=/home/ubuntu/data/eth3d_raw
for s in courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains; do
  n=$(ls $RAW/$s/ground_truth_depth/dslr_images 2>/dev/null | wc -l)
  if [ "$n" -lt 5 ]; then
    echo "== $s: downloading depth"
    (cd $RAW && curl -sS -L -o ${s}_depth.7z https://www.eth3d.net/data/${s}_dslr_depth.7z && 7z x -y ${s}_depth.7z > /dev/null; rm -f ${s}_depth.7z)
  fi
  echo "== $s: preprocess"
  python scripts/eth3d.py $RAW /home/ubuntu/data/eth3d --scenes $s 2>&1 | grep -v pynvml
  rm -rf $RAW/$s
  df -h / | tail -1
done
echo ETH3D_PREP_DONE
