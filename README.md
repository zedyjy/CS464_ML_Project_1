# CS464_ML_Project_1

To download dataset:
    mkdir train test   
    pip install awscli
    cd train
    aws s3 --no-sign-request cp s3://rareplanes-public/real/tarballs/train/RarePlanes_train_geojson_aircraft_tiled.tar.gz .
    aws s3 --no-sign-request cp s3://rareplanes-public/real/tarballs/train/RarePlanes_train_PS-RGB_tiled.tar.gz .
    cd ../test
    aws s3 --no-sign-request cp s3://rareplanes-public/real/tarballs/test/RarePlanes_test_geojson_aircraft_tiled.tar.gz . 
    aws s3 --no-sign-request cp s3://rareplanes-public/real/tarballs/test/RarePlanes_test_PS-RGB_tiled.tar.gz .
    pip install pandas matplotlib geopandas opencv-python-headless tqdm
    
