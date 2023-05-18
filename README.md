# Demo webpage for recognizing bricks


Currently recognizes 447 popular bricks. Photo must be taken
on a white background.


## Setup

```
# install python deps
python -m pip install -r requirements.txt

# render icons for bricks with lego-rendering -> ./run.sh render-line-art.py
# then copy from renders/line-art to this folder ./static/images

# download classification model, too big for Github
wget https://brian-lego-public.s3.us-west-1.amazonaws.com/03-447x.pt -o 03-447x.pt
```


## Running

```
python serve.py
./ngrok http 8000  # if you want to access outside your computer
```

This repo includes a copy of the ngrok binary. I wasn't able to use
it from npm but probably an issue with my machine
