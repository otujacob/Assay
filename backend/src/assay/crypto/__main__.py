"""Open or rewrap a sealed file: `python -m assay.crypto open|rewrap ...` (FR-41).

  python -m assay.crypto open assay-audit.csv.sealed --tenant tenant-a -o assay-audit.csv
  python -m assay.crypto rewrap assay-audit.csv.sealed --tenant tenant-a --key-version 2

The key comes from ASSAY_MASTER_KEY (the local key provider, see docs/security/key-management.md).
Anyone who holds that secret can open every tenant's files, so run this only where that is intended.
A sealed audit export is bound to the tenant and to the context "audit-export", so it will not open
for another tenant, and a file sealed for something else will not open here.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from . import CryptoError, LocalKeyProvider, key_version, open_sealed, rewrap

DEFAULT_CONTEXT = "audit-export"


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    ap = argparse.ArgumentParser(prog="python -m assay.crypto", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, helptext in (("open", "decrypt a sealed file"), ("rewrap", "move a sealed file to a newer key version, in place")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("file", type=Path)
        p.add_argument("--tenant", required=True)
        p.add_argument("--context", default=DEFAULT_CONTEXT, help=f"default: {DEFAULT_CONTEXT}")
    sub.choices["open"].add_argument("-o", "--output", type=Path, help="write here instead of standard output")
    sub.choices["open"].add_argument("--force", action="store_true", help="overwrite the output file")
    sub.choices["rewrap"].add_argument("--key-version", type=int, required=True,
                                       help="the key version to move the file to")
    args = ap.parse_args(argv)

    master = env.get("ASSAY_MASTER_KEY")
    if not master:
        print("ASSAY_MASTER_KEY is not set", file=sys.stderr)
        return 1
    try:
        blob = args.file.read_bytes()
    except OSError as e:
        print(f"cannot read {args.file}: {e.strerror}", file=sys.stderr)
        return 1
    try:
        if args.cmd == "open":
            provider = LocalKeyProvider(master.encode())
            data = open_sealed(provider, args.tenant, blob, args.context)
            if args.output is None:
                sys.stdout.buffer.write(data)
            elif args.output.exists() and not args.force:
                print(f"{args.output} exists; use --force to overwrite", file=sys.stderr)
                return 1
            else:
                args.output.write_bytes(data)
            return 0
        provider = LocalKeyProvider(master.encode(), versions={args.tenant: args.key_version})
        was = key_version(blob)
        new = rewrap(provider, args.tenant, blob, args.context)
        if new == blob:
            print(f"already on key version {was}")
            return 0
        tmp = args.file.with_name(args.file.name + ".tmp")
        tmp.write_bytes(new)
        tmp.replace(args.file)  # atomic: the file is never half-written
        print(f"rewrapped from key version {was} to {args.key_version}")
        return 0
    except CryptoError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
