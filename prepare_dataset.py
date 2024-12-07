import os

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
                with open(song_file_name, 'r', encoding="utf-8") as file:
                    content = file.read()
                    if len(content) == 0:
                        continue
                    lyrics = LyricsPreprocessor.preprocess(content)
                    lyrics = lyrics.replace('\n', ' ')
                    dataset_filename.write(lyrics)
                    dataset_filename.write('\n')


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

prepare_dataset(artists, "dataset.txt")