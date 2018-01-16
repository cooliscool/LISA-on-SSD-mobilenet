# LISA-on-SSD-mobilenet
For training SSD MobileNet v1 Trained on LISA Traffic Sign dataset.

The code in this repository is helpful to Convert the LISA Traffic Sign dataset into Tensorflow tfrecords.

## How To Use
1. Download LISA Dataset here : http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html
2. Specify dataset root directory and Output directory

Example usage:
```
    ./create_lisa_tf_record --data_dir=/home/user/lisa \
        --output_dir=/home/user/lisa/output
        
```    
## Credits
Code adapted and modified by Ajmal Moochingal
Original code credit goes to Tensorflow Authors. 
