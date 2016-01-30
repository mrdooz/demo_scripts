# shader compile script

import os
import time
import glob
import subprocess
from collections import OrderedDict, defaultdict
from string import Template
import re
import inc_bin
import argparse

ENTRY_POINT_TAG = 'entry-point'
SHADER_DECL_RE = re.compile('(.+)? (.+)\(.*')
ENTRY_POINT_RE = re.compile('// entry-point: (.+)')
FULL_SCREEN_ENTRY_POINT_RE = re.compile('// full-screen-entry-point: (.+)')
CBUFFER_META_BEGIN_RE = re.compile('// cb-meta-begin: (.+)')
CBUFFER_META_END_RE = re.compile('// cb-meta-end')
CBUFFER_RANGE_RE = re.compile('// (.+): range: (.+)\.\.(.+)')
CBUFFER_TYPE_RE = re.compile('// (.+): type: (.+)')
# cbuffer cbRadialGradient : register(b1)
CBUFFER_RE = re.compile('cbuffer (.+) : .+')
DEPS_RE = re.compile('#include "(.+?)"')

# for each file, contain list of entry-points by shader type
SHADERS = {}
SHADER_FILES = set()
DEPS = defaultdict(set)
LAST_FAIL_TIME = {}

SHADER_DIR = None
OUT_DIR = None

SHADER_DATA = {
    'vs': {'profile': 'vs', 'obj_ext': 'vso', 'asm_ext': 'vsa'},
    'gs': {'profile': 'gs', 'obj_ext': 'gso', 'asm_ext': 'gsa'},
    'ps': {'profile': 'ps', 'obj_ext': 'pso', 'asm_ext': 'psa'},
    'cs': {'profile': 'cs', 'obj_ext': 'cso', 'asm_ext': 'csa'},
}

# conversion between HLSL and my types
KNOWN_TYPES = {
    'float': {'type': 'float', 'size': 1},
    'float2': {'type': 'vec2', 'size': 2},
    'float3': {'type': 'vec3', 'size': 3},
    'float4': {'type': 'vec4', 'size': 4},
    'float4x4': {'type': 'mtx4x4', 'size': 16},
    'matrix': {'type': 'mtx4x4', 'size': 16},
}

CBUFFER_NAMESPACE = None
CBUFFER_TEMPLATE = Template("""#pragma once
namespace $namespace
{
  namespace cb
  {
$cbuffers
  }
}
""")


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def parse_hlsl_file(f):
    # scan the hlsl file, and look for:
    # - entry points
    # - dependencies
    # - cbuffer meta
    res = defaultdict(list)
    deps = set()
    cbuffer_meta = defaultdict(dict)
    cbuffer_shader = None
    parse_cbuffer_header = False
    _, filename = os.path.split(f)
    entry_point_type = None
    full_screen_entry_point = None
    for r in open(f, 'rt').readlines():
        r = r.strip()

        if parse_cbuffer_header:
            # the previous row was a cbuffer meta end, so we parse the cbuffer
            # definition now to get the name
            m = CBUFFER_RE.match(r)
            if m:
                cbuffer_meta[cbuffer_shader]['cbuffer'] = m.groups()[0]
            parse_cbuffer_header = False
            cbuffer_shader = None
            continue

        # Check for c-buffer meta data
        if cbuffer_shader:
            if CBUFFER_META_END_RE.match(r):
                parse_cbuffer_header = True
                continue

            m = CBUFFER_RANGE_RE.match(r)
            if m:
                g = m.groups()
                cbuffer_meta[cbuffer_shader][g[0]]['range'] = [
                    float(g[1]), float(g[2])]

            m = CBUFFER_TYPE_RE.match(r)
            if m:
                g = m.groups()
                cbuffer_meta[cbuffer_shader][g[0]]['type'] = g[1]
            continue

        m = CBUFFER_META_BEGIN_RE.match(r)
        if m:
            # associate this cbuffer with a shader, and create the cbuffer
            # dict
            cbuffer_shader = m.groups()[0]
            cbuffer_meta[cbuffer_shader] = defaultdict(dict)

        elif entry_point_type:
            # previous row was an entry point, so parse the entry point
            # name
            m = SHADER_DECL_RE.match(r)
            if m:
                name = m.groups()[1]
                res[entry_point_type].append((name, full_screen_entry_point))
            entry_point_type = None
            full_screen_entry_point = None
        else:
            m0 = ENTRY_POINT_RE.match(r)
            m1 = FULL_SCREEN_ENTRY_POINT_RE.match(r)
            full_screen_entry_point = m1 is not None
            if m0 or m1:
                m = m0 if m0 else m1
                t = m.groups()[0]
                if t in ('vs', 'ps', 'gs', 'cs'):
                    # found correct entry point tag
                    entry_point_type = t
                else:
                    print 'Unknown tag type: %s' % t
            else:
                m = DEPS_RE.match(r)
                if m:
                    deps.add(m.groups()[0])

    return res, deps, cbuffer_meta


def generate_filenames(base, entry_points, obj_ext, asm_ext):
    # returns the output files from the given base and entry points
    res = []
    for entry_point in entry_points:
        e, is_fullscreen = entry_point
        res.append((base + '_' + e + '.' + obj_ext, e, False))
        res.append((base + '_' + e + '.' + asm_ext, e, False))
        res.append((base + '_' + e + 'D.' + obj_ext, e, True))
        res.append((base + '_' + e + 'D.' + asm_ext, e, True))
    return res


def parse_cbuffer(basename, asm_filename):
    """ Parses the asm-file, and collects the cbuffer variables """

    cbuffer_prefix = basename.title().replace('.', '')

    cbuffers = []
    cur_cbuffer = None
    cur_input_sig = None
    try:
        with open(asm_filename) as f:
            lines = f.readlines()
    except:
        print 'file not found parsing cbuffers: %s' % asm_filename
        return

    skip_count = 0
    for line in lines:
        if skip_count:
            skip_count -= 1
            continue

        if not line.startswith('//'):
            continue
        line = line[3:]
        line = line.strip()

        if line.startswith('cbuffer'):
            name = line[len('cbuffer '):]
            cur_cbuffer = {
                'name': cbuffer_prefix + name,
                'root': name,
                'vars': OrderedDict(),
                'unused': 0,
            }
            continue
        elif line.startswith('}'):
            cbuffers.append(cur_cbuffer)
            cur_cbuffer = None
            continue
        elif line.startswith('Input signature:'):
            cur_input_sig = True
            skip_count = 3
            continue

        if cur_input_sig:
            if not line:
                # done with current input sig
                cur_input_sig = None
                continue
            else:
                line = line.split()
                # the 'use' field isn't always set (for sys values?)
                used = None
                if len(line) == 6:
                    name, index, mask, register, sys_value, format = line
                elif len(line) == 7:
                    name, index, mask, register, sys_value, format, used = line

        if not cur_cbuffer:
            continue

        tmp, _, comments = line.partition(';')
        comments = comments.strip()
        if comments.find('[unused]') != -1:
            cur_cbuffer['unused'] += 1
        var_type, _, var_name = tmp.partition(' ')
        if not var_type or not var_name:
            continue
        if var_type not in KNOWN_TYPES:
            continue
        cur_cbuffer['vars'][var_name] = (KNOWN_TYPES[var_type], comments)

    return cbuffers


def save_cbuffer(cbuffer_filename, cbuffers):
    """ write the cbuffers to the given header file as a struct """
    num_valid = 0
    bufs = []
    for c in cbuffers:
        name = c['name']
        vars = c['vars']

        # skip writing the cbuffer if all the vars are unused
        if len(vars) == c['unused']:
            continue
        num_valid += 1

        cur = '    struct %s\n    {\n' % name

        # calc max line length to align the comments
        max_len = 0
        for n, (var_data, comments) in vars.iteritems():
            t = var_data['type']
            max_len = max(max_len, len(n) + len(t))

        padder = 0
        slots_left = 4
        for n, (var_data, comments) in vars.iteritems():
            var_type = var_data['type']
            var_size = var_data.get('size', None)
            # if the current variable doesn't fit in the remaining slots,
            # align it
            if (slots_left != 4 and slots_left - var_size < 0):
                cur += (
                    '      float padding%s[%s];\n' %
                    (padder, slots_left)
                )
                padder += 1
                slots_left = 4
            cur_len = len(n) + len(var_type)
            padding = (max_len - cur_len + 8) * ' '
            cur += '      %s %s;%s%s\n' % (var_type, n, padding, comments)
            slots_left -= (var_size % 4)
            if slots_left == 0:
                slots_left = 4
        cur += '    };'

        bufs.append(cur)

    if num_valid:
        res = CBUFFER_TEMPLATE.substitute({
            'cbuffers': '\n'.join(bufs),
            'namespace': CBUFFER_NAMESPACE,
        })

        # check if this is identical to the previous cbuffer
        identical = False
        try:
            with open(cbuffer_filename, 'rt') as f:
                identical = f.read() == res
        except IOError:
            pass

        if not identical:
            with open(cbuffer_filename, 'wt') as f:
                f.write(res)


def save_manifest(root, cbuffers, cbuffer_meta):
    # the manifest contains information about all the (pixel) shaders in a
    # hlsl file, along with the cbuffer data. the idea is that using this
    # data, we can automatically load shaders and create imgui displays for
    # them.
    manifest_file = os.path.join(OUT_DIR, root + '.manifest')
    with open(manifest_file, 'wt') as f:
        for shader in SHADERS.get(root, {}).get('ps', []):
            shader, is_fullscreen = shader
            # only write manifests for fullscreen shaders with cbuffer meta
            # data
            meta = cbuffer_meta.get(shader)
            if not is_fullscreen or not meta:
                continue
            f.write('shader-begin name: %s file: %s\n' % (
                shader, os.path.join(OUT_DIR, root + '_' + shader + '.pso')))

            f.write('cbuffer-begin\n')
            # for the current shader, find the cbuffer with associated meta
            cur_cbuffer = cbuffer_meta[shader]['cbuffer']

            for cbs in cbuffers.get(shader, {}):
                if cbs['root'] != cur_cbuffer:
                    continue

                for key, value in cbs.get('vars', {}).iteritems():
                    # look for metadata associated with the current var
                    if key in meta:
                        if 'type' in meta[key]:
                            val_type = meta[key]['type']
                        else:
                            val_type = value[0]['type']

                        if 'range' in meta[key]:
                            m = meta[key]
                            f.write(
                                'name: %s type: %s min: %s max: %s\n' % (
                                    key, val_type,
                                    m['range'][0], m['range'][1]))
                        else:
                            f.write('name: %s type: %s\n' % (key, val_type))
                    else:
                        f.write(
                            'name: %s type: %s\n' % (key, value[0]['type']))
            f.write('cbuffer-end\n')
            f.write('shader-end\n')


def should_compile(full_path):
    # first step is to see if the shader has a .status file - if it doesn't
    # then it's never been compiled
    path, root = os.path.split(full_path)
    status_file = os.path.join(OUT_DIR, root + '.status')
    if not os.path.exists(status_file):
        return True

    # status = open(status_file, 'rt').readline().strip()
    status_date = os.path.getmtime(status_file)
    hlsl_date = os.path.getmtime(full_path)

    # if the hlsl file has any deps, check the dates of those too
    for dep in DEPS[root]:
        full_dep = os.path.join(path, dep)
        hlsl_date = max(hlsl_date, os.path.getmtime(full_dep))

    return status_date < hlsl_date


def compile(full_path, root, cbuffer_meta):
    # shader_file =  ..\shaders\landscape.landscape
    shader_file = os.path.join(SHADER_DIR, root)
    hlsl_file = shader_file + '.hlsl'
    hlsl_file_time = os.path.getmtime(hlsl_file)
    status_file = os.path.join(OUT_DIR, root + '.hlsl.status')
    with open(status_file, 'wt') as f:
        f.write('OK\n')

    cbuffers = {}

    for shader_type, entry_points in SHADERS[root].iteritems():
        profile = SHADER_DATA[shader_type]['profile']
        obj_ext = SHADER_DATA[shader_type]['obj_ext']
        asm_ext = SHADER_DATA[shader_type]['asm_ext']

        # compile all the entry points
        g = generate_filenames(root, entry_points, obj_ext, asm_ext)
        for output, entry_point, is_debug in g:
            out_root = os.path.join(OUT_DIR, root + '_' + entry_point)
            suffix = 'D' if is_debug else ''
            obj_file = out_root + suffix + '.' + obj_ext
            asm_file = out_root + suffix + '.' + asm_ext

            if is_debug:
                # create debug shader
                # returns 0 on success, > 0 otherwise
                res = subprocess.call([
                    'fxc',
                    '/nologo',
                    '/T%s_5_0' % profile,
                    '/Od',
                    '/Zi',
                    '/E%s' % entry_point,
                    '/Fo%s' % (obj_file),
                    '/Fc%s' % (asm_file),
                    '%s.hlsl' % shader_file])
            else:
                # create optimized shader
                res = subprocess.call([
                    'fxc',
                    '/nologo',
                    '/T%s_5_0' % profile,
                    '/O3',
                    '/E%s' % entry_point,
                    '/Fo%s' % (obj_file),
                    '/Fc%s' % (asm_file),
                    '/Qstrip_debug',
                    '/Qstrip_reflect',
                    '/Vd',
                    '%s.hlsl' % shader_file])
            if res:
                # if compilation failed, don't try again until the
                # .hlsl file has been updated
                print '** FAILURE: %s, %s' % (shader_file, entry_point)
                LAST_FAIL_TIME[shader_file] = hlsl_file_time
            else:
                if not is_debug:
                    inc_bin.dump_bin(obj_file)

                if is_debug:
                    cb = parse_cbuffer(root, asm_file)
                    if shader_type == 'ps':
                        cbuffers[entry_point] = cb
                    if CBUFFER_NAMESPACE:
                        cbuffer_filename = (out_root + '.cbuffers.hpp').lower()
                        save_cbuffer(cbuffer_filename, cb)

                if shader_file in LAST_FAIL_TIME:
                    del(LAST_FAIL_TIME[shader_file])

    save_manifest(root, cbuffers, cbuffer_meta)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--shader-dir', default='shaders')
parser.add_argument('-o', '--out-dir', default='out')
parser.add_argument('-bin', '--binary', action='store_true')
parser.add_argument(
    '-cb', '--constant-buffer', help='Constant buffer namespace')
args = parser.parse_args()
SHADER_DIR = args.shader_dir
OUT_DIR = os.path.join(args.shader_dir, args.out_dir)
CBUFFER_NAMESPACE = args.constant_buffer

try:
    prev_files = set()
    while True:
        safe_mkdir(OUT_DIR)
        cur_files = set()
        first_tick = True
        for full_path in glob.glob(os.path.join(SHADER_DIR, '*.hlsl')):
            _, filename = os.path.split(full_path)
            root, ext = os.path.splitext(filename)
            cur_files.add(root)
            if should_compile(full_path):
                # If first modified file, print header
                if first_tick:
                    first_tick = False
                    ll = time.localtime()
                    print '==> COMPILE STARTED AT [%.2d:%.2d:%.2d]' % (
                        ll.tm_hour, ll.tm_min, ll.tm_sec)
                # the hlsl file has changed, so reparse it's deps and entry
                # points
                entry_points, deps, cbuffer_meta = parse_hlsl_file(full_path)
                SHADERS[root] = entry_points
                DEPS[root] = deps

                for row in open(full_path).readlines():
                    m = DEPS_RE.match(row)
                    if m:
                        dependant = m.groups()[0]
                        DEPS[root].add(dependant)

                compile(full_path, root, cbuffer_meta)

        # purge data from files that no longer exist
        for root in prev_files.difference(cur_files):
            if root in SHADERS:
                del SHADERS[root]
            if root in DEPS:
                del DEPS[root]

        prev_files = cur_files

        time.sleep(1)
except KeyboardInterrupt:
    print 'Exiting'
    exit(1)
