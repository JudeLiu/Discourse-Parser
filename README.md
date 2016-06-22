# Components

- model/: contains trained models
- lib/: contains necessary files
- dict/: contains useful files used for training
- classifier.py: classifier
- tester.py: predict test set
- other files

# How to Use

## prerequisite

- Python 2.7
- nltk
- numpy
- Berkeley parser
- Stanford parser

Python 2.x is needed. nltk and numpy are two python packages that can be downloaded with the help of various tools. In my case, I use pip to manage python packages. Therefore type the command

```
    pip install numpy
    pip insatll nltk
```

to download and install numpy and nltk.

Berkeley parser and Stanford parser are stored in lib/ directory so you do not have to worry about it.

## How to Predict?

Simply type the command

```
    python tester.py [-p] [-d] test_file_name
```

Then it will output a predict json file named test\_file\_name\_predict.json, and it is stored in current working directory. 

-p -d parameters are optional. It is used when you want to predict a test set that has been predicted before. More specifically, before predicting, the program extracts production rules and dependency rules and stores them in the tmp/ directory. So if you predict the very test set again, you do not have to extract these features again, since they are stored in the hard disk physically. Thus in this case, type

```
    python tester.py -p -d test_file_name
```