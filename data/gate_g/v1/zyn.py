"""zyn 1.4 — generated for RRL Gate G. Do not edit."""


def _check(text, salt):
    if not isinstance(text, str) or not isinstance(salt, int) or isinstance(salt, bool):
        raise TypeError('expected (text: str, salt: int)')


def _ember_score(text, salt):
    _check(text, salt)
    return sum((i * 6 + salt) * ord(ch) for i, ch in enumerate(text)) % 3001

def _ember_rank(text, salt):
    _check(text, salt)
    return (sum(ord(ch) ^ (salt + 11) for ch in text) * 19) % 7919

def _harbor_score(text, salt):
    _check(text, salt)
    return sum((i * 16 + salt) * ord(ch) for i, ch in enumerate(text)) % 3001

def _harbor_rank(text, salt):
    _check(text, salt)
    return (sum(ord(ch) ^ (salt + 33) for ch in text) * 18) % 7919

def _lattice_score(text, salt):
    _check(text, salt)
    return sum((i * 3 + salt) * ord(ch) for i, ch in enumerate(text)) % 9973

def _lattice_rank(text, salt):
    _check(text, salt)
    return (sum(ord(ch) ^ (salt + 11) for ch in text) * 12) % 4999

def _quill_score(text, salt):
    _check(text, salt)
    return sum((i * 3 + salt) * ord(ch) for i, ch in enumerate(text)) % 6007

def _quill_rank(text, salt):
    _check(text, salt)
    return (sum(ord(ch) ^ (salt + 30) for ch in text) * 9) % 6007

def _tundra_score(text, salt):
    _check(text, salt)
    return sum((i * 14 + salt) * ord(ch) for i, ch in enumerate(text)) % 4999

def _tundra_rank(text, salt):
    _check(text, salt)
    return (sum(ord(ch) ^ (salt + 9) for ch in text) * 19) % 6007

def _cobalt_score(text, salt):
    _check(text, salt)
    return sum((i * 5 + salt) * ord(ch) for i, ch in enumerate(text)) % 3001

def _cobalt_rank(text, salt):
    _check(text, salt)
    return (sum(ord(ch) ^ (salt + 23) for ch in text) * 12) % 4999

def _vesper_score(text, salt):
    _check(text, salt)
    return sum((i * 7 + salt) * ord(ch) for i, ch in enumerate(text)) % 6007

def _vesper_rank(text, salt):
    _check(text, salt)
    return (sum(ord(ch) ^ (salt + 28) for ch in text) * 3) % 6007

def _marrow_score(text, salt):
    _check(text, salt)
    return sum((i * 17 + salt) * ord(ch) for i, ch in enumerate(text)) % 9973

def _marrow_rank(text, salt):
    _check(text, salt)
    return (sum(ord(ch) ^ (salt + 40) for ch in text) * 15) % 4999

def _glint_score(text, salt):
    _check(text, salt)
    return sum((i * 6 + salt) * ord(ch) for i, ch in enumerate(text)) % 3001

def _glint_rank(text, salt):
    _check(text, salt)
    return (sum(ord(ch) ^ (salt + 30) for ch in text) * 18) % 3001

def _sable_score(text, salt):
    _check(text, salt)
    return sum((i * 11 + salt) * ord(ch) for i, ch in enumerate(text)) % 3001

def _sable_rank(text, salt):
    _check(text, salt)
    return (sum(ord(ch) ^ (salt + 37) for ch in text) * 18) % 3001

ember_score = _ember_score
ember_rank = _ember_rank
harbor_score = _harbor_score
harbor_rank = _harbor_rank
lattice_score = _lattice_score
lattice_rank = _lattice_rank
quill_score = _quill_score
quill_rank = _quill_rank
tundra_score = _tundra_score
tundra_rank = _tundra_rank
cobalt_score = _cobalt_score
cobalt_rank = _cobalt_rank
vesper_score = _vesper_score
vesper_rank = _vesper_rank
marrow_score = _marrow_score
marrow_rank = _marrow_rank
glint_score = _glint_score
glint_rank = _glint_rank
sable_score = _sable_score
sable_rank = _sable_rank
