"""Keep a reusable pairing URL in a private file, outside the checkout."""
import os
import secrets
from urllib.parse import parse_qs, urlsplit, urlunsplit


def pairing_token(path, public_url):
    parsed = urlsplit(public_url)
    if parsed.scheme not in ('ws', 'wss') or not parsed.hostname or parsed.query or parsed.fragment or parsed.username or parsed.password or parsed.path != '/control':
        raise ValueError('Use a ws(s)://host[:port]/control URL without credentials or query')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = urlsplit(path.read_text().strip())
        query = parse_qs(previous.query, keep_blank_values=True)
        values = query.get('token', [])
        if (urlunsplit(previous._replace(query='', fragment='')) == public_url
                and set(query) == {'token'} and len(values) == 1
                and len(values[0]) >= 32
                and all(c.isascii() and (c.isalnum() or c in '-_') for c in values[0])):
            path.chmod(0o600)
            return values[0]
    token = secrets.token_urlsafe(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(path, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(public_url + '?token=' + token + '\n')
    return token
