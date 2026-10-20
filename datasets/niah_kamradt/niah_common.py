"""Shared corpus + haystack machinery for the NIAH-Kamradt and RULER NIAH-8
generators (see generate_niah_kamradt.py).

Design contract (both datasets):
  * Haystack filler = the canonical Paul Graham essay corpus used by both
    Kamradt's LLMTest_NeedleInAHaystack and NVIDIA RULER's essay mode
    (fetched once from HF `sgoel9/paul_graham_essays`, cached locally).
  * Essays are split 80/20 into TRAIN-haystack vs TEST-haystack pools by a
    stable md5 hash of the title, so test fillers are never seen in training.
  * Needle KEYS (cities for Kamradt; adj-noun / uuid keys for RULER) are split
    train/test by the same stable-hash rule — the train set never contains a
    needle whose query key appears in the test set.  This is the whole
    decontamination story: every needle fact is generated, so there is no
    external corpus overlap to purge (unlike the wiki-based twins).
  * Token sizing uses the Qwen3-1.7B tokenizer (byte-identical to 8B).
    Sentences are tokenized ONCE and haystacks are assembled from cached
    per-sentence token counts, so generation is tokenizer-bound only at load.

Depth convention: depth d in [0,1] inserts the needle at the sentence boundary
whose cumulative token count is closest to d * total_tokens (d=0 → very first
sentence, d=1 → appended at the end), matching Kamradt's insertion rule.
"""
import hashlib
import json
import os
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PG_CACHE = Path(os.environ.get("NIAH_DATA_DIR", _HERE)) / "pg_essays.jsonl"
MODEL_TOK = "Qwen/Qwen3-1.7B"   # sizing tokenizer (byte-identical to Qwen3-8B)

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def is_test_key(s: str, test_mod: int = 5) -> bool:
    """Deterministic ~20% test membership by content hash."""
    return stable_hash("split::" + s) % test_mod == 0


def load_pg_essays():
    """Return [{title, text}] for the 215-essay PG corpus, caching to jsonl."""
    if PG_CACHE.exists():
        with PG_CACHE.open() as f:
            rows = [json.loads(l) for l in f if l.strip()]
        if len(rows) < 200:
            raise RuntimeError(f"{PG_CACHE} exists but has only {len(rows)} rows; delete and re-fetch")
        return rows
    from datasets import load_dataset
    ds = load_dataset("sgoel9/paul_graham_essays", split="train")
    rows = [{"title": r["title"], "text": " ".join(str(r["text"]).split())} for r in ds]
    if len(rows) < 200:
        raise RuntimeError(f"PG essay fetch returned {len(rows)} rows (<200) — wrong dataset?")
    with PG_CACHE.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def split_sentences(text: str):
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


class EssayPool:
    """Pre-tokenized sentence store over one split's essays.

    assemble(rng, target_tokens) returns (sentences, tok_counts): whole essays
    in rng-shuffled order, the last one trimmed at a sentence boundary so the
    total lands at (or just under) target_tokens — coherent local doc flow,
    Kamradt-style, with per-sample diversity from the essay order.
    """

    def __init__(self, essays, tokenizer):
        self.essays = []           # list[list[(sentence, n_tok)]]
        for e in essays:
            sents = split_sentences(e["text"])
            if len(sents) < 5:
                continue
            enc = tokenizer([" " + s for s in sents], add_special_tokens=False)["input_ids"]
            self.essays.append(list(zip(sents, [len(x) for x in enc])))
        if not self.essays:
            raise RuntimeError("EssayPool: no usable essays")

    def assemble(self, rng, target_tokens: int):
        order = list(range(len(self.essays)))
        rng.shuffle(order)
        out_s, out_n, total = [], [], 0
        for ei in order:
            for s, n in self.essays[ei]:
                if total + n > target_tokens:
                    if total < 0.98 * target_tokens:
                        continue          # small sentence may still fit later
                    return out_s, out_n
                out_s.append(s); out_n.append(n); total += n
        return out_s, out_n               # corpus exhausted (never at 16k targets)


def insert_at_depth(sentences, tok_counts, needle: str, needle_tok: int, depth: float):
    """Insert `needle` at the sentence boundary closest to depth*total tokens.
    Returns (sentences, tok_counts, actual_depth)."""
    total = sum(tok_counts)
    if total == 0:
        raise ValueError("empty haystack")
    target = depth * total
    cum = 0
    best_i, best_d = 0, abs(0 - target)
    for i, n in enumerate(tok_counts):
        cum += n
        d = abs(cum - target)
        if d < best_d:
            best_i, best_d = i + 1, d
    s2 = sentences[:best_i] + [needle] + sentences[best_i:]
    n2 = tok_counts[:best_i] + [needle_tok] + tok_counts[best_i:]
    pre = sum(tok_counts[:best_i])
    return s2, n2, pre / total


# ── needle-key component lists (embedded; wonderwords not in the env) ─────────
# ~250 world cities for the Kamradt needle ("The best thing to do in {city} …").
CITIES = """Tokyo Delhi Shanghai Dhaka Cairo Beijing Mumbai Osaka Karachi Chongqing
Istanbul Kolkata Manila Lagos Guangzhou Shenzhen Kinshasa Lahore Bangalore Moscow
Tianjin Jakarta London Lima Bogota Chennai Bangkok Hyderabad Nagoya Chengdu
Tehran Nanjing Wuhan Ahmedabad Hangzhou Seoul Santiago Riyadh Surat Madrid
Suzhou Pune Harbin Houston Dallas Toronto Miami Belem Brasilia Atlanta
Singapore Fukuoka Barcelona Johannesburg Berlin Bekasi Nairobi Hanoi Sydney Melbourne
Luanda Alexandria Guadalajara Monterrey Ankara Recife Montreal Porto Fortaleza Medan
Durban Casablanca Nagpur Algiers Kabul Amman Havana Rome Kyiv Caracas
Vienna Munich Milan Prague Dublin Brussels Lisbon Warsaw Budapest Amsterdam
Zurich Geneva Oslo Stockholm Copenhagen Helsinki Athens Bucharest Sofia Belgrade
Zagreb Sarajevo Tallinn Riga Vilnius Krakow Gdansk Granada Valencia Seville
Bilbao Marseille Lyon Toulouse Nice Nantes Strasbourg Bordeaux Lille Rennes
Rotterdam Utrecht Antwerp Ghent Bruges Cardiff Glasgow Edinburgh Manchester Liverpool
Leeds Bristol Oxford Cambridge York Bath Nottingham Sheffield Newcastle Belfast
Auckland Wellington Christchurch Brisbane Perth Adelaide Hobart Darwin Canberra Suva
Honolulu Anchorage Seattle Portland Denver Phoenix Tucson Albuquerque Boise Reno
Sacramento Oakland Pasadena Monterey Vancouver Calgary Edmonton Ottawa Quebec Winnipeg
Halifax Victoria Saskatoon Regina Nassau Kingston Bridgetown Georgetown Paramaribo Quito
Guayaquil Cusco Arequipa Valparaiso Mendoza Cordoba Rosario Montevideo Asuncion Salvador
Curitiba Manaus Maputo Windhoek Gaborone Lusaka Harare Kampala Kigali Accra
Abuja Dakar Bamako Tunis Tripoli Khartoum Djibouti Mogadishu Antananarivo Zanzibar
Muscat Doha Kuwait Manama Beirut Damascus Baghdad Basra Shiraz Isfahan
Tashkent Samarkand Almaty Astana Bishkek Dushanbe Ashgabat Baku Tbilisi Yerevan
Kathmandu Thimphu Colombo Male Dhulikhel Yangon Mandalay Vientiane Phnompenh Hue
Danang Taipei Kaohsiung Kyoto Sapporo Sendai Hiroshima Nara Kanazawa Okinawa
Busan Incheon Daegu Ulaanbaatar Irkutsk Novosibirsk Vladivostok Kazan Samara Sochi""".split()

# Activity components for Kamradt needles: core + venue + condition → phrase.
ACT_CORES = [
    "eat a sandwich", "fly a paper kite", "sketch the skyline", "read a slim novel",
    "share a pot of tea", "watch the fishing boats", "listen to street musicians",
    "photograph the old doorways", "sample the market cheeses", "feed the pigeons",
    "browse the secondhand bookstalls", "paint a tiny watercolor", "eat warm chestnuts",
    "drink a slow espresso", "count the passing trams", "write postcards",
    "taste the local pastries", "watch the lantern lighting", "join a dawn swim",
    "rent a rickety bicycle", "hunt for vinyl records", "watch the tide turn",
    "sip cold lemonade", "trace the old city walls", "follow the mural trail",
    "picnic on rye bread and plums", "learn three words of the dialect",
    "watch the kite surfers", "queue for dumplings", "collect fallen ginkgo leaves",
    "listen to the evening bells", "watch the glassblowers work", "row a wooden skiff",
    "eat grilled corn", "study the tilework", "watch the night market open",
    "climb the clocktower stairs", "search for sea glass", "watch the storks nest",
    "share roasted peanuts",
]
ACT_VENUES = [
    "in the old botanical garden", "along the riverfront promenade", "on the harbor breakwater",
    "under the railway arches", "in the covered spice market", "beside the fountain square",
    "on the museum steps", "along the canal towpath", "in the sculpture courtyard",
    "on the hilltop overlook", "near the lighthouse point", "in the university quadrangle",
    "beneath the fig trees", "on the ferry upper deck", "at the flower auction hall",
    "in the tiled courtyard", "by the tidal pools", "along the city ramparts",
    "in the lantern alley", "at the open-air amphitheater", "on the pedestrian bridge",
    "in the cactus conservatory", "beside the rose pergola", "at the fishermen's quay",
    "in the clock museum garden", "under the observatory dome", "at the kite field",
    "on the cathedral plaza", "in the birch grove park", "at the old tram depot",
    "near the salt warehouses", "in the mosaic arcade", "on the floating dock",
    "at the botanical maze", "in the printmakers' lane", "by the reservoir dam",
    "at the terraced tea garden", "in the amber gallery district", "on the grand staircase",
    "near the puppet theater",
]
ACT_CONDS = [
    "on a sunny day", "at first light", "just after the rain", "in the late afternoon",
    "as the fog lifts", "during the golden hour", "on a breezy morning", "under a full moon",
    "before the shops open", "when the bells ring noon", "as the swallows return",
    "on a crisp autumn evening", "while the mist holds", "at low tide",
    "during the spring festival", "on a quiet Sunday", "as the lamps come on",
    "in the first snow", "when the jacarandas bloom", "at the height of summer",
]

# The literal canonical Kamradt needle, reserved for the test split.
CANON_CITY = "San Francisco"
CANON_ACTIVITY = "eat a sandwich and sit in Dolores Park on a sunny day"
CANON_QUESTION = "What is the best thing to do in San Francisco?"

# Adjective/noun pools for RULER word keys ("{adj}-{noun}").
RULER_ADJS = """amber ancient azure bold brisk calm cedar cheerful copper coral crimson curious
dappled dusty eager early emerald faded fearless fierce gentle gilded gleaming golden
granite hazel hidden hollow humble icy indigo ivory jade jolly keen kindly lively lonely
lucky lunar marble mellow mighty misty modest mossy nimble noble northern olive opal
pale patient pearl plum polished proud quiet rapid restless rosy rugged rustic sable
sacred sandy scarlet serene shady silent silver sleepy solemn spry stately steady stormy
sturdy sunlit swift tawny thorny tidal timber tranquil trusty tundra twilight umber
upland velvet verdant vivid wandering weathered western wild windy wintry woven zesty""".split()
RULER_NOUNS = """acorn anchor anvil archway atlas badger banner barrel beacon bellows birch bison
bramble brook cairn caldera canyon caravan cascade cavern cedar chapel chimney cinder
citadel cliff cobble comet compass coral cottage crag creek crescent crow current delta
dune eagle ember estuary falcon fjord flint forge fossil fountain gale garnet geyser
glacier glade gorge granary grotto grove gully harbor hearth heather hollow horizon
iceberg inlet island juniper kestrel knoll lagoon lantern larch ledge lighthouse
lynx marsh meadow mesa mill moor moraine mosaic moss obelisk orchard osprey otter
outpost oxbow parapet pebble pinnacle plateau prairie quarry quill rampart rapids raven
reef ridge rill riverbed rookery rune saddle sagebrush sandbar shale shoal sparrow
spire spring steppe summit sundial swale tarn terrace thicket tide timberline trellis
tributary tundra turret vale valley vault vineyard walnut waterfall weir willow windmill""".split()
