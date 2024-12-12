
class LyricsPreprocessor:
    MARKER_END_OF_LINE = None
    MARKER_SONG_NAME_START = None
    MARKER_SONG_NAME_END = None
    MARKER_SONG_START = None
    MARKER_SONG_END = None

    @staticmethod
    def replace_characters(lyrics):
        return (lyrics.replace('е', 'e')
                .replace(u'\u03bc', 'mu'))

    @staticmethod
    def preprocess(lyrics, title):
        lyrics = LyricsPreprocessor.replace_characters(lyrics)
        lyrics = lyrics.replace('\n', LyricsPreprocessor.MARKER_END_OF_LINE)
        lyrics = f"{LyricsPreprocessor.MARKER_SONG_NAME_START}{title}{LyricsPreprocessor.MARKER_SONG_NAME_END}{LyricsPreprocessor.MARKER_SONG_START}{lyrics}{LyricsPreprocessor.MARKER_SONG_END}"
        return lyrics


LyricsPreprocessor.MARKER_END_OF_LINE = '{END_OF_LINE}'
LyricsPreprocessor.MARKER_SONG_NAME_START = '{SONG_NAME_START}'
LyricsPreprocessor.MARKER_SONG_NAME_END = '{SONG_NAME_END}'
LyricsPreprocessor.MARKER_SONG_START = '{SONG_START}'
LyricsPreprocessor.MARKER_SONG_END = '{SONG_END}'