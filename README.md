# Synthetic Countenance
_An exploration of faces within the [PGAN](https://github.com/tkarras/progressive_growing_of_gans) model_

### Getting started

** Setup **
+ Make sure you have a working GPU version of tensorflow installed.
+ Install the dependencies `pip install -r requirements.txt`
+ Download the model file from [Google Drive](https://drive.google.com/open?id=188K19ucknC6wg1R6jbuPEhTq9zoufOx4) and place in the [/model](/model) directory.

** Samples and features **
+ Create some samples `python P0_generate_samples.py` or `make samples`
+ Look at the data in samples/images/
+ Generate features with `make features`

## Dev NOTES

    python 3.6.0
    tensorflow-gpu==1.12.3
