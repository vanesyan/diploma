Required Ubuntu

System deps
```
$ apt install tesseract-ocr
```

Python deps
```
$ pip3 install -r requirements.txt
```

To test program program itself:
```
$ python3 generate_tests.py
```

To test program on synthetic tests run:
```
usage: node_detection.py [-h] [--handwritten] [--scale SCALE] [--debug]
                         [--maxdiff MAXDIFF]
                         path

Optical block function diagram recognition.

positional arguments:
  path               Path to file to process

optional arguments:
  -h, --help         show this help message and exit
  --handwritten      Indicates handwritten mode
  --scale SCALE      Sets image scale factor
  --debug            Debug mode
  --maxdiff MAXDIFF  Indicates maximum diff pixels that allowed to be
                     considered a known shape
```

If program is lagged try to rescale image, if it is too wide.
If you'd like to test program on handwritten schemas provide `--handwritten` flag.