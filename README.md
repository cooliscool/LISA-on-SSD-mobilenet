# LISA-on-SSD-mobilenet
For training a SSD using MobileNet-v1 by Google with LISA Traffic Sign dataset.

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

**this project is archived. I don't work on this anymore. but, anyone can feel free to use it or modify or do whatever you want.**
