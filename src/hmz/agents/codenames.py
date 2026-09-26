"""What an agent nobody named is called, which is a codename out of Amphoreus.

An agent given no name still needs one nothing else answers to, and `ClaudeCodeAgent#3f2a1b9c`
is a name only a debugger loves. So one is drawn from the Chrysos Heirs instead -- the golden
blooded of Amphoreus, whose codes *Honkai: Star Rail* spells as a Greek word for what the heir
was made to be and three digits with something behind them. `NeiKos496` is strife, and 496 is
the third perfect number; `SkeMma720` is inquiry, and 720 is 6!; `KaLos618` is beauty, and 618
is the golden ratio. The era the story is told in is the 33,550,336th, which is the fifth
perfect number, so the numbers are the joke and not decoration.

Twelve of the thirteen carry a code the story says out loud. The thirteenth is the outsider
who walked into the simulation and was never issued one, which is the shape of this file: the
twelve are the canon, and everything else is another era of the same roles under another
number -- Amphoreus reruns its heirs, and two of them have already shared `EpieiKeia216` across
eras. The canon comes up far more often than its share would give it, a name being only a
joke to somebody who recognises it.

The rule builds words rather than only listing them. A Greek word is morphemes joined at a
capital -- `Apo` and `Ria` are `ApoRia`, which is an heir's -- so `MetaKratos`, `PolyMorphe`
and `NeoTelos` are words the same rule makes and the story merely never needed. That is what
makes the space endless: when the short words are used up the next one grows a morpheme, so
there is always another code and never a tail of hex to fall back to. `Chaoz666`, the
misnumbered serial out of an early era, is the one code that breaks the rule, and this
generates it no more than the story explains it.
"""

from __future__ import annotations

import itertools
import random
import threading

__all__ = ["codename"]

HEIRS: tuple[tuple[str, str], ...] = (
    ("NeiKos", "496"),  
    ("PoleMos", "600"),  
    ("SkeMma", "720"),  
    ("EpieiKeia", "216"),  
    ("HapLotes", "405"),  
    ("KaLos", "618"),  
    ("EleOs", "252"),  
    ("HubRis", "504"),  
    ("PhiLia", "093"),  
    ("ApoRia", "432"),  
    ("OreXis", "945"),  
    ("SkoPeo", "365"),  
)

WORDS: tuple[str, ...] = (
    "AgaPe",
    "AiDos",
    "AiOn",
    "AiTher",
    "AisThesis",
    "AletheIa",
    "AnamNesis",
    "AnanKe",
    "AndreIa",
    "AneMos",
    "ApeiRon",
    "ArChe",
    "AreTe",
    "ArithMos",
    "AsTer",
    "AtaraXia",
    "AthaNasia",
    "AutarKeia",
    "BiOs",
    "ChaRis",
    "ChroNos",
    "DaiMon",
    "DeiMos",
    "DemiOurgos",
    "DiKe",
    "DoLos",
    "DoXa",
    "DynaMis",
    "EidOs",
    "ElPis",
    "EnerGeia",
    "EpisTeme",
    "ErIs",
    "EtHos",
    "EudaiMonia",
    "GaIa",
    "GeneSis",
    "GnoSis",
    "HarmoNia",
    "HedoNe",
    "HeimarMene",
    "HeLios",
    "HemeRa",
    "HenoSis",
    "HoLos",
    "HorKos",
    "HyDor",
    "HyLe",
    "HypNos",
    "KaiRos",
    "KanOn",
    "KatharSis",
    "KenoMa",
    "KineSis",
    "KleOs",
    "KleRos",
    "KosMos",
    "KriSis",
    "KykLos",
    "LeThe",
    "LoGos",
    "LyPe",
    "MeLos",
    "MeTis",
    "MetRon",
    "MiasMa",
    "MimeSis",
    "MneMe",
    "MoiRa",
    "MonAs",
    "MorPhe",
    "NemeSis",
    "NoMos",
    "NosTos",
    "NoUs",
    "NykTos",
    "OiKos",
    "OrGe",
    "OuraNos",
    "OuSia",
    "ParrheSia",
    "PaThos",
    "PenThos",
    "PhaOs",
    "PhanTasia",
    "PhoBos",
    "PhosPhoros",
    "PhroNesis",
    "PhthoRa",
    "PisTis",
    "PleRoma",
    "PneuMa",
    "PoieSis",
    "PoiNe",
    "PoLis",
    "PoNos",
    "PraXis",
    "ProNoia",
    "PseuDos",
    "PsyChe",
    "RhythMos",
    "SeLene",
    "SoPhia",
    "SophroSyne",
    "SoTer",
    "StaSis",
    "StorGe",
    "SymBolon",
    "TechNe",
    "TeLos",
    "ThalasSa",
    "ThanaTos",
    "TheMis",
    "TheoRia",
    "ThyMos",
    "TiMai",
    "TriAs",
    "TyChe",
    "XeNia",
    "XeNos",
)

JOINS: tuple[str, ...] = (
    "Amphi",
    "Ana",
    "Anti",
    "Apo",
    "Archi",
    "Auto",
    "Dia",
    "Dys",
    "Ek",
    "En",
    "Endo",
    "Epi",
    "Eu",
    "Exo",
    "Hemi",
    "Hetero",
    "Homo",
    "Hyper",
    "Hypo",
    "Iso",
    "Kata",
    "Makro",
    "Meso",
    "Meta",
    "Mikro",
    "Mono",
    "Neo",
    "Ortho",
    "Oxy",
    "Palin",
    "Pan",
    "Para",
    "Peri",
    "Poly",
    "Pro",
    "Proto",
    "Pseudo",
    "Syn",
    "Tele",
    "Tetra",
    "Tri",
)

STEMS: tuple[str, ...] = (
    "Aion",
    "Arche",
    "Bios",
    "Chronos",
    "Doxa",
    "Genesis",
    "Gnosis",
    "Grapha",
    "Kosmos",
    "Kratos",
    "Krisis",
    "Latreia",
    "Lexis",
    "Logos",
    "Machia",
    "Mania",
    "Metron",
    "Moira",
    "Morphe",
    "Nomos",
    "Nous",
    "Odos",
    "Pathos",
    "Phonia",
    "Phora",
    "Pistis",
    "Pneuma",
    "Poiesis",
    "Praxis",
    "Psyche",
    "Rhythmos",
    "Skopos",
    "Soma",
    "Sophia",
    "Sphaira",
    "Stasis",
    "Taxis",
    "Techne",
    "Telos",
    "Thesis",
    "Thymos",
    "Tonos",
    "Topos",
    "Tropos",
    "Tyche",
)

JOINED = 2

CANON = 0.5

AGAIN = 0.5

KNOWN = 0.5

_CALLED: set[str] = set()
_CALLING = threading.Lock()

_COUNTING = itertools.count(1)

_TRIES = 64

def codename() -> str:
    """Draws a codename no agent in this process has been given.

    Returns:
      The code, as a Greek word and three digits -- `NeiKos496`, `KykLos204`, `MetaKratos881`.
    """
    with _CALLING:
        for _ in range(_TRIES):
            drawn = _drawn()
            if drawn not in _CALLED:
                _CALLED.add(drawn)
                return drawn
        return _counted()

def _drawn() -> str:
    """One draw, canon-heavy, which may be a code already given out.

    The canon half draws from the heirs nobody holds yet rather than from all twelve, so the
    first agents a process leaves unnamed get whole heirs instead of losing half their draws
    to a code already out. Once the twelve are gone that half falls through to the rest, which
    is what Amphoreus does with them anyway: the role runs again under another number.

    Returns:
      A code, canon or otherwise.
    """
    if random.random() < CANON:  
        canon = (f"{word}{number}" for word, number in HEIRS)
        if free := [one for one in canon if one not in _CALLED]:
            return random.choice(free)  
    return f"{_word()}{random.randrange(1000):03d}"  

def _word() -> str:
    """A word to hang three digits off: an heir's, one Greek keeps, or one built to order.

    Returns:
      The word, capitalised at the front and where it breaks.
    """
    if random.random() < AGAIN:  
        return random.choice(HEIRS)[0]  
    if random.random() < KNOWN:  
        return random.choice(WORDS)  
    return random.choice(JOINS) + random.choice(STEMS)  

def _counted() -> str:
    """A code built to order, for a process that has drawn most of what luck can find.

    Counted rather than drawn, so it always answers and always by the rule. The count is read
    off as morphemes -- the stem lowest, the joins above it -- which is a place notation over
    the two lists, so every count spells a different word and the word grows one morpheme each
    time the shorter ones run out. There is no last code, and so nothing to fall back to: an
    agent nobody named has a Greek word and three digits however many have been handed out.

    Returns:
      The first counted code nobody holds.
    """
    while True:
        count, number = divmod(next(_COUNTING), 1000)
        count, index = divmod(count, len(STEMS))
        parts = [STEMS[index]]
        while count or len(parts) < JOINED:
            count, index = divmod(count, len(JOINS))
            parts.append(JOINS[index])
        drawn = f"{''.join(reversed(parts))}{number:03d}"
        if drawn not in _CALLED:
            _CALLED.add(drawn)
            return drawn
