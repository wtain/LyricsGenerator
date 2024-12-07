import os
import re
from collections import Counter

from Environment import Environment
from LyricsGenius.lyricsgenius.genius import Genius
from LyricsPreprocessor import LyricsPreprocessor


def save_lyrics(song):

    def canonize_filename(filename):
        return re.sub(r'[^a-zA-Z0-9 \n.]', '', filename)

    def split_words(lyrics):
        rgx = re.compile(r"([\w][\w']*\w)")
        return rgx.findall(lyrics)

    artist = song.artist
    title = song.title

    filename_stats = f"./lyrics_stats/{canonize_filename(artist)}/{canonize_filename(title)}.txt"
    filename_lyrics = f"./lyrics/{canonize_filename(artist)}/{canonize_filename(title)}.txt"
    os.makedirs(os.path.dirname(filename_stats), exist_ok=True)
    os.makedirs(os.path.dirname(filename_lyrics), exist_ok=True)

    with open(filename_lyrics, 'w', encoding="utf-8") as file:
        file.write(song.lyrics)

    lyrics = LyricsPreprocessor.preprocess(song.lyrics)

    words = Counter(split_words(lyrics))
    stats = "\n".join(f"{word}: {words[word]}" for word in sorted(words, key=lambda w: -words[w]))
    with open(filename_stats, 'w', encoding="utf-8") as file:
        file.write(stats)

genius = Genius(Environment.read_token(), sleep_time=0.05)

def process_artist(artist):
    artist = genius.search_artist(artist, allow_name_change=False)
    for song in artist.songs:
        print(f"Processing {artist} - {song.title}")
        save_lyrics(song)


artists = [
    "Children of Bodom",
    "Bodom After Midnight",
    "Sinergy",
    "Kalmah",
    "Norther",
    "Skyfire",
    "Mors Principium Est",
    "Nekrogoblikon",
    "Dimmu Borgir",
    "System of a Down",
    "Slayer",
    "Eternal Tears Of Sorrow",
    "Iron Maiden",
    "Manowar",
]

for artist in artists:
    process_artist(artist)
