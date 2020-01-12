# TATIA

Classify a tweet set.

Report generated with the following command.

```
pandoc --pdf-engine=xelatex -V CJKmainfont="Source Code Pro" rapport.md -o rapport.pdf -N -V colorlinks -V urlcolor=NavyBlue
```

## Dependencies

- Python 3.8+
- `python-nltk`

```
$ python -m pip --user install nltk
```

## Usage

```
usage: classification.py [-h] [-t TWEETS]

Classify a tweet set

optional arguments:
  -h, --help            show this help message and exit
  -t TWEETS, --tweets TWEETS
                        Path to the tweet dataset to analyse
```

The entry point is `classification.py`, which takes a bit of time
to run (so expect to wait a few minutes).

By default, it analyses the dataset at `data/twitter-2016devtest-BD.tsv`.

You can provide a custom dataset using the `-t` argument.

```
$ python classification.py -t data/twitter-2016test-BD.tsv
```

The format is expected to follow this convention:

```
tweet id	user name	expected status	tweet
```

Here, every whitespace is a tab character (`\t`).
