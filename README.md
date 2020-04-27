# SlitScan

## Setup

### compile module

```
$ python setup.py build_ext --inplace
```


## How to use

### slit scan
```
$  ./main.py --timeshift-type random
```

### random pixel delay
```
$ ./main.py --timeshift-type slitscan
```

### with recorded video file
```
$ ./main.py video_file_name
```
