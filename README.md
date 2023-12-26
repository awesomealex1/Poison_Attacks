# Poison Attacks

To download FaceForensics++ data, run ```scripts/download_ff_data.sh``` (these will be videos). It is necessary to first get access to the FaceForensics++ download script from the official FaceForensics++ github repository. Once you have access to it, place it in the scripts directory like this: ```scripts/faceforensics_download_v4.py```

To convert FaceForensics++ videos to images, run ```run_parallel_jobs.sh``` from the ```/scripts``` folder.

To download XceptionNet, run ```scripts/download_xception.sh```

To run a feature collision attack on pretrained XceptionNet and FF++ data, run ```pipelines/xception_ff_feature_collision.py```
