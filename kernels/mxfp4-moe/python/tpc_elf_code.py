#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""sha256 of the machine code of TPC kernel ELFs: the .text and .tpc_metadata sections (build.sh writes it to MANIFEST).

    tpc_elf_code.py KERNEL.o              one ELF: prints the hash
    tpc_elf_code.py --lib libmxk_tpc.so   every kernel ELF embedded in a kernel library: prints "<index> <hash>"

tpc-clang embeds the source text, the command line and the build directory in the ELF, so the hash of the whole
file changes with a comment or a path. These two sections do not: equal hashes mean the same kernel code.
"""

import hashlib
import struct
import sys

CODE_SECTIONS = (".text", ".tpc_metadata")


def code_hash(b):
    """sha256 over the code sections of one little-endian ELF32 image starting at b[0]."""
    shoff = struct.unpack_from("<I", b, 0x20)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x2E)
    hdrs = [struct.unpack_from("<IIIIII", b, shoff + i * shentsize) for i in range(shnum)]
    stro = hdrs[shstrndx][4]
    sec = {}
    for name_off, typ, _flags, _addr, off, size in hdrs:
        name = b[stro + name_off : b.index(b"\0", stro + name_off)].decode()
        sec[name] = b[off : off + size] if typ != 8 else b""  # SHT_NOBITS has no file bytes
    h = hashlib.sha256()
    for n in CODE_SECTIONS:
        h.update(n.encode() + b"\0" + sec[n])
    return h.hexdigest()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--lib":
        with open(sys.argv[2], "rb") as f:
            data = f.read()
        i, k = 0, 0
        while (i := data.find(b"\x7fELF\x01\x01", i + 1)) >= 0:
            try:
                print(k, code_hash(data[i:]))
                k += 1
            except (struct.error, KeyError, ValueError):
                pass  # an ELF magic inside other data
        return
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with open(sys.argv[1], "rb") as f:
        print(code_hash(f.read()))


if __name__ == "__main__":
    main()
