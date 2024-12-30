import os
import re

from Environment import Environment
from LyricsGenius.lyricsgenius.genius import Genius

from Commons import Commons

def save_lyrics(song):

    def canonize_filename(filename):
        return re.sub(r'[^a-zA-Z0-9 \n.]', '', filename)

    artist = song.artist
    title = song.title

    filename_lyrics = f"./lyrics/{canonize_filename(artist)}/{canonize_filename(title)}.txt"
    os.makedirs(os.path.dirname(filename_lyrics), exist_ok=True)

    with open(filename_lyrics, 'w', encoding="utf-8") as file:
        file.write(song.lyrics)

genius = Genius(Environment.read_token(), sleep_time=0.01, timeout=25)

def process_artist(artist):
    artist = genius.search_artist(artist, allow_name_change=False)
    for song in artist.songs:
        print(f"Processing {artist} - {song.title}")
        save_lyrics(song)
    # try:
    #     artist = genius.search_artist(artist, allow_name_change=False)
    #     for song in artist.songs:
    #         print(f"Processing {artist} - {song.title}")
    #         save_lyrics(song)
    # except Exception as e:
    #     print(f"Error: {e}")


for artist in Commons.artists:
    process_artist(artist)
