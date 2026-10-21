# TODO

- **Bump the locked certifi after the eMudhra CA compromise**: The private keys of eight eMudhra intermediate CAs were compromised ([incident summary](https://emudhra.we-have-a-problem.com/)), so anyone holding them could impersonate any HTTPS endpoint to clients that still trust them. `uv.lock` pins certifi 2026.1.4, which still trusts those intermediates through the emSign ECC Root CA - G3. Josh's local `.venv` was upgraded to certifi 2026.10.19, but `uv sync` would downgrade it again. Next: run `uv lock --upgrade-package certifi` and sync.
