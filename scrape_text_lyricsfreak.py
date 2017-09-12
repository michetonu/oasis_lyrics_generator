from bs4 import BeautifulSoup
from urllib.request import urlopen, build_opener
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import textmining
import lda
import numpy as np
import string
from googletrans import Translator
import pickle
import os
from fake_useragent import UserAgent

ua = UserAgent()


def get_links(url, band_name):
    page_url = os.path.join(url, band_name[0], band_name)
    opener = build_opener()
    opener.addheaders = [('User-agent', ua.random)]
    response = opener.open(page_url)  # Open the nth page of Together
    page = response.read()

    links = []
    soup = BeautifulSoup(page, "lxml")  # Get HTML text

    for thread in soup.find_all('a'):  # Find all links to single threads and save them
        if thread.parent.name == 'td':
            link_name = thread.get('href')
            if link_name[:2] == '/' + band_name[0]:
                link_name = url + link_name
                links.append(link_name)
    print('Links extracted')
    return links


def get_text(links):
    songs = []
    i = 1

    for link in links:
        opener = build_opener()
        opener.addheaders = [('User-agent', ua.random)]
        response = opener.open(link)  # Open the nth page of Together
        page = response.read()
        soup_song = BeautifulSoup(page, "lxml")

        for result in soup_song.find_all('div', attrs={'id': 'content_h', 'class': 'dn'}):
            a = result.get_text()
            words = a.split()
            for i in range(len(words)):
                if words[i] == words[i+1]:
                    del words[i+1]
            words = " ".join(sorted(set(words), key=words.index))
            print(words)
            songs.append(words)

        i += 1
        print('Song #{} extracted'.format(i))
    return songs


if __name__ == "__main__":
    url = "http://www.lyricsfreak.com"
    band_name = 'oasis'
    links = get_links(url, band_name)
    songs = get_text(links)
    with open("oasis_lyrics.pckl", 'wb') as ff:
        pickle.dump(songs, ff)