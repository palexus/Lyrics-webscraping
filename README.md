# Text Classification

This project was made to learn the basics of webscraping and text classification. It scrapes the lyrics of your favorite bands (lyrics.com) and then you can type in a text and it will tell you to which band it probably belongs.

The focus lied on the scraping. The model for text classification is therefore very simple.


# How to use it?

Clone this project and create a new environement, e.g.

 `conda create -n webscraping -c conda-forge scikit-learn`.
 
 Then activate your new environment, here it is called webscraping and install the requirements.
 
`conda activate webscraping`

`pip install -r requirements.txt`
 
 
### If you run it for the first time,

the you need to download the stopwords. We us here nltk and the english stopwords. If you use a different language you have to change this.

`python stopwords_loader.py`

### Run the program

The program is used in the Command-line. Run the main file and use your bands as attributes, e.g. 

`python text_classification.py "Scooter" "Nirvana"`

will download - if not already done - the lyrics of Scooter and Nirvana and compare them.
