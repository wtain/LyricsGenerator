import re

from unicodedata import normalize


class LyricsPreprocessor:

    @staticmethod
    def clean_annotations(lyrics):
        cleaned = re.sub(r'\[.*]', '', lyrics, flags=re.MULTILINE)
        cleaned = re.sub(r'\(Solo:.*\)', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\(chorus\)', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^Solo:.*$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^-.*-$', '', cleaned, flags=re.MULTILINE)
        return cleaned

    @staticmethod
    def clean_blank_lines(lyrics):
        return "\n".join(filter(lambda x: not re.match(r'^\s*$', x), lyrics.split('\n')))

    @staticmethod
    def filter_allowed_characters(lyrics):
        return "".join(filter(lambda c: str.isalpha(c) or c == '\'' or str.isspace(c), lyrics))

    @staticmethod
    def decapitalize(lyrics):
        return "".join(map(lambda c: str.lower(c), lyrics))

    @staticmethod
    def remove_special_unicode_chars(lyrics):
        return normalize('NFKD', lyrics)

    @staticmethod
    def replace_characters(lyrics):
        return (lyrics.replace('е', 'e')
                .replace(u'\u03bc', 'mu'))

    @staticmethod
    def preprocess(lyrics):
        lyrics = LyricsPreprocessor.clean_annotations(lyrics)
        lyrics = LyricsPreprocessor.clean_blank_lines(lyrics)
        lyrics = LyricsPreprocessor.replace_characters(lyrics)
        lyrics = LyricsPreprocessor.filter_allowed_characters(lyrics)
        lyrics = LyricsPreprocessor.decapitalize(lyrics)
        return LyricsPreprocessor.remove_special_unicode_chars(lyrics)