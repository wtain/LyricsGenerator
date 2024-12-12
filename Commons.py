
from LyricsPreprocessor import LyricsPreprocessor

class Commons:
    artists = None
    special_tokens = None


Commons.artists = [
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
    "Metallica",
    "Imperanon",
    "Frozen Crown",
    "Bullet for My Valentine",
    "Alestorm",
    "Instorm",
    "In Flames",
    "Dark Tranquillity",
    "At The Gates",
    "Amorphis",
    "Megadeth",
    "Death",
    "Noumena",
    "Amon Amarth",
    "Arch Enemy",
    "Slipknot",
    "Avril Lavigne",
    "Wintersun",
    "Insomnium",
    "Omnium Gatherum",
    "Ensiferum",
    "The Black Dahlia Murder",
    "Cryhavoc",
    "Anthrax",
    "Megadeth",
    "Nox Aeterna",
    "Opeth",
    "Naildown",
    "Frozen Crown",
    "Aephanemer",
    "Pantera",
    "Dethklok",
    "Be'lakor",
]

Commons.special_tokens = {
    "additional_special_tokens": [
        LyricsPreprocessor.MARKER_END_OF_LINE,
        LyricsPreprocessor.MARKER_SONG_NAME_START,
        LyricsPreprocessor.MARKER_SONG_NAME_END,
        LyricsPreprocessor.MARKER_SONG_START,
        LyricsPreprocessor.MARKER_SONG_END,
    ]
}