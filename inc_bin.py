from glob import glob
import sys
import itertools
import os


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)


def dump_bin(f):
    res = []
    for chunk in grouper(open(f, 'rb').read(), 16):
        res.append(
            ', '.join(["0x%.2x" % ord(x) for x in chunk if x is not None]))
    res = ',\n'.join(res)

    _, tail = os.path.split(f)
    filename, _ = os.path.splitext(tail)
    inc = 'u8 %s_bin[] = {\n%s};\n' % (filename, res)
    with open(os.path.abspath(f) + '.hpp', 'w') as out_file:
        out_file.write(inc)


if __name__ == '__main__':
    # inputs are a list of globs
    for arg in sys.argv[1:]:
        for f in glob(arg):
            dump_bin(f)
