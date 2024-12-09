import os

import Commons
from LyricsPreprocessor import LyricsPreprocessor


def prepare_dataset(artists, output_file_name):
    try:
        os.remove(output_file_name)
    except:
        pass
    with open(output_file_name, 'a', encoding="utf-8") as dataset_filename:
        for artist in artists:
            artist_directory = f"./lyrics/{artist}"
            for song_file in os.listdir(artist_directory):
                song_file_name = f"{artist_directory}/{song_file}"
                title = os.path.basename(song_file_name).split(".")[0]
                with open(song_file_name, 'r', encoding="utf-8") as file:
                    content = file.read()
                    if len(content) == 0:
                        continue
                    lyrics = LyricsPreprocessor.preprocess(content, title)
                    lyrics = lyrics.replace('\n', ' ')
                    dataset_filename.write(lyrics)
                    dataset_filename.write('\n')


prepare_dataset(Commons.artists, "dataset.txt")