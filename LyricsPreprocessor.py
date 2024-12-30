
class LyricsPreprocessor:
    MARKER_END_OF_LINE = None
    MARKER_SONG_NAME_START = None
    MARKER_SONG_NAME_END = None
    MARKER_SONG_START = None
    MARKER_SONG_END = None
    PAD_TOKEN = None
    bos_token = None
    eos_token = None
    unk_token = None

    @staticmethod
    def replace_characters(lyrics):
        return (lyrics.replace('е', 'e')
                .replace(u'\u03bc', 'mu'))

    @staticmethod
    def preprocess(lyrics, title):
        lyrics = LyricsPreprocessor.replace_characters(lyrics)
        lyrics = lyrics.replace('\n', LyricsPreprocessor.MARKER_END_OF_LINE)
        # lyrics = f"{LyricsPreprocessor.MARKER_SONG_NAME_START}{title}{LyricsPreprocessor.MARKER_SONG_NAME_END}{LyricsPreprocessor.MARKER_SONG_START}{lyrics}{LyricsPreprocessor.MARKER_SONG_END}"
        lyrics = f"{LyricsPreprocessor.bos_token}{LyricsPreprocessor.MARKER_SONG_NAME_START}{title}{LyricsPreprocessor.MARKER_SONG_NAME_END}{LyricsPreprocessor.MARKER_END_OF_LINE}{lyrics}{LyricsPreprocessor.eos_token}"
        return lyrics


LyricsPreprocessor.MARKER_END_OF_LINE = '<|linebreak|>'
LyricsPreprocessor.MARKER_SONG_NAME_START = '<|songnamestart|>'
LyricsPreprocessor.MARKER_SONG_NAME_END = '<|songnameend|>'
LyricsPreprocessor.MARKER_SONG_START = '<|lyricsstart|>'
LyricsPreprocessor.MARKER_SONG_END = '<|lyricsend|>'
LyricsPreprocessor.PAD_TOKEN = '<|pad|>'
LyricsPreprocessor.bos_token = '<|startoftext|>'
LyricsPreprocessor.eos_token = '<|endoftext|>'
LyricsPreprocessor.unk_token = '<|unknown|>'