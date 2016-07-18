"""shader compile script."""

import os
import time
import glob
import subprocess
from collections import OrderedDict, defaultdict
from string import Template
import re
import inc_bin
import argparse

FXC_PATH = 'C:/Program Files (x86)/Windows Kits/8.1/bin/x86/fxc.exe'
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
    'int': {'type': 'int', 'size': 1},
    'uint': {'type': 'u32', 'size': 1},
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

CBUFFER_TEMPLATE_RAW = Template("""#pragma once
namespace cb
{
$cbuffers
}
""")

FULLSCREEN_CBUFFER = """cbuffer cbFullscreen : register(c0)
{
    float2 g_dim;
    float2 g_time;
};
"""

CBUFFER_HLSL_TEMMPLATE = Template("""cbuffer $name : register(c$register)
{
$vars
};
""")


def replace_ext(full_path, new_ext):
    # NB, new_ext doesn't include the '.'
    root, _ = os.path.splitext(full_path)
    return root + '.' + new_ext


def _safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def _parse_hlsl_file(f):
    # scan the hlsl file, and look for:
    # - entry points
    # - dependencies
    # - cbuffer meta
    entry_points_per_type = defaultdict(list)
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
                entry_points_per_type[entry_point_type].append(
                    (name, full_screen_entry_point))
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
                    print('Unknown tag type: %s' % t)
            else:
                m = DEPS_RE.match(r)
                if m:
                    deps.add(m.groups()[0])

    return entry_points_per_type, deps, cbuffer_meta


def _parse_meta_file(filename):
    """
    Parse the meta file, and create a cbuffer file
    """

    seen_keys = set()
    cur_var = {}
    cbuffer_vars = []
    cbuffers = []

    for line in open(filename).readlines():
        line = line.strip()
        if not line or line[0] == '#':
            continue

        # all meta lines are of the form: 'key : value'
        k, _, v = line.partition(':')
        v = v.strip()

        if k == 'fullscreen' and v.lower() == 'true':
            # fullscreen effects need an addition cbuffer with dims/time
            cbuffers.append(FULLSCREEN_CBUFFER)
            continue

        if k in seen_keys:
            # second time we see a key, so this is a new cbuffer
            cbuffer_vars.append(cur_var)
            cur_var = {}
            seen_keys = set()

        seen_keys.add(k)
        cur_var[k] = v

    cbuffer_vars.append(cur_var)

    # create mapping for any types where the c++ and hlsl type names don't
    # agree
    type_mapping = {
        'color': 'float4'
    }

    # create all the rows in the cbuffer
    rows = []
    max_len = 0
    for r in cbuffer_vars:
        var_type = type_mapping.get(r['type'], r['type'])
        cur = '%s %s;' % (var_type, r['name'])
        rows.append(cur)
        max_len = max(max_len, len(cur))

    # add the row comments, using the max_len to align them nicely :)
    for i, r in enumerate(rows):
        diff = max_len - len(r)
        rows[i] = (
            '    ' + r + ' ' * (4 + diff) +
            '// ' + cbuffer_vars[i].get('comment', ''))

    cbuffers.append(CBUFFER_HLSL_TEMMPLATE.substitute(
        name='cbMeta',
        register=len(cbuffers),
        vars='\n'.join(rows)))

    # write the cbuffer file
    with open(replace_ext(filename, 'cbuffers.hlsl'), 'wt') as f:
        f.write('\n'.join(cbuffers))


def _generate_filenames(base, entry_points, obj_ext, asm_ext):
    # returns the output files from the given base and entry points
    res = []
    for entry_point in entry_points:
        e, is_fullscreen = entry_point
        res.append((base + '_' + e + '.' + obj_ext, e, False))
        res.append((base + '_' + e + '.' + asm_ext, e, False))
        res.append((base + '_' + e + 'D.' + obj_ext, e, True))
        res.append((base + '_' + e + 'D.' + asm_ext, e, True))
    return res


def _parse_cbuffer(basename, asm_filename):
    """
    Parses the asm-file, and collects the cbuffer variables
    """

    cbuffer_prefix = basename.title().replace('.', '')

    cbuffers = []
    cur_cbuffer = None
    cur_input_sig = None
    try:
        with open(asm_filename) as f:
            lines = f.readlines()
    except:
        print('file not found parsing cbuffers: %s' % asm_filename)
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
            if cur_cbuffer:
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
        # NB: we don't do any kind of filtering on lines, so this will match
        # '{' etc, which should be fine, as this won't match a valid type..
        # print 'line: %s\ntype: %s, name: %s' % (line, var_type, var_name)
        if not var_type or not var_name:
            continue

        # check if the type is an array type
        array_size = 1
        if '[' in var_name and ']' in var_name:
            open_idx = var_name.find('[')
            close_idx = var_name.find(']')
            array_size = int(var_name[open_idx+1:close_idx])
            var_name = var_name[:open_idx]
        if var_type not in KNOWN_TYPES:
            continue

        cur_cbuffer['vars'][var_name] = {
            'type': KNOWN_TYPES[var_type]['type'],
            'size': KNOWN_TYPES[var_type]['size'],
            'array_size': array_size,
            'comments': comments
        }

    return cbuffers


def _save_cbuffer(cbuffer_filename, cbuffers):
    """Write the cbuffers to the given header file as a struct."""
    num_valid = 0
    bufs = []
    indent = 4 * ' ' if CBUFFER_NAMESPACE else 2 * ' '
    for c in cbuffers:
        name = c['name']
        vars = c['vars']

        # skip writing the cbuffer if all the vars are unused
        if len(vars) == c['unused']:
            continue
        num_valid += 1

        cur = '%sstruct %s\n    {\n' % (indent, name)

        # calc max line length to align the comments
        max_len = 0
        for n, var_data in vars.iteritems():
            t = var_data['type']
            max_len = max(max_len, len(n) + len(t))

        padder = 0
        slots_left = 4
        for var_name, var_data in vars.iteritems():
            var_type = var_data['type']
            var_size = var_data['size']
            array_size = var_data['array_size']
            comments = var_data['comments']

            # adjust var size, and var name based on array size
            if array_size > 1:
                var_size *= array_size
                var_name = '%s[%d]' % (var_name, array_size)

            # if the current variable doesn't fit in the remaining slots,
            # align it
            if (slots_left != 4 and slots_left - var_size < 0):
                cur += (
                    '%sfloat padding%s[%s];\n' %
                    (indent, padder, slots_left)
                )
                padder += 1
                slots_left = 4
            cur_len = len(var_name) + len(var_type)
            padding = (max_len - cur_len + 8) * ' '
            cur += (
                '      %s %s;%s%s\n' % (var_type, var_name, padding, comments))
            slots_left -= (var_size % 4)
            if slots_left == 0:
                slots_left = 4
        cur += '%s};' % (indent)

        bufs.append(cur)

    if num_valid:
        if CBUFFER_NAMESPACE:
            res = CBUFFER_TEMPLATE.substitute({
                'cbuffers': '\n'.join(bufs),
                'namespace': CBUFFER_NAMESPACE,
            })
        else:
            res = CBUFFER_TEMPLATE_RAW.substitute({
                'cbuffers': '\n'.join(bufs)
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


def _save_manifest(root, cbuffers, cbuffer_meta):
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


def _should_compile(full_path):
    # first step is to see if the shader has a .status file - if it doesn't
    # then it's never been compiled
    path, root = os.path.split(full_path)
    status_file = os.path.join(OUT_DIR, root + '.status')
    if not os.path.exists(status_file):
        return True

    status_date = os.path.getmtime(status_file)
    hlsl_date = os.path.getmtime(full_path)

    # if the hlsl file has any deps, check the dates of those too
    for dep in DEPS[root]:
        full_dep = os.path.join(path, dep)
        hlsl_date = max(hlsl_date, os.path.getmtime(full_dep))

    # if there's an assoicated .meta file, check that date too
    meta_file = replace_ext(full_path, 'meta')
    if os.path.exists(meta_file):
        hlsl_date = max(hlsl_date, os.path.getmtime(meta_file))

    return status_date < hlsl_date


def _compile(full_path, root, cbuffer_meta):
    # shader_file =  ..\shaders\landscape.landscape
    shader_file = os.path.join(SHADER_DIR, root)
    hlsl_file = shader_file + '.hlsl'
    hlsl_file_time = os.path.getmtime(hlsl_file)
    status_file = os.path.join(OUT_DIR, root + '.hlsl.status')
    with open(status_file, 'wt') as f:
        f.write('OK\n')

    cbuffers = {}
    compile_res = {'ok': set(), 'fail': set()}

    for shader_type, entry_points in SHADERS[root].iteritems():
        profile = SHADER_DATA[shader_type]['profile']
        obj_ext = SHADER_DATA[shader_type]['obj_ext']
        asm_ext = SHADER_DATA[shader_type]['asm_ext']

        # compile all the entry points
        g = _generate_filenames(root, entry_points, obj_ext, asm_ext)
        for output, entry_point, is_debug in g:
            out_root = os.path.join(OUT_DIR, root + '_' + entry_point)
            suffix = 'D' if is_debug else ''
            obj_file = out_root + suffix + '.' + obj_ext
            asm_file = out_root + suffix + '.' + asm_ext

            if is_debug:
                # create debug shader
                # returns 0 on success, > 0 otherwise
                res = subprocess.call([
                    FXC_PATH,
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
                    FXC_PATH,
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
                print('** FAILURE: %s, %s' % (shader_file, entry_point))
                LAST_FAIL_TIME[shader_file] = hlsl_file_time
                compile_res['fail'].add(entry_point)
            else:
                compile_res['ok'].add(entry_point)
                if not is_debug:
                    inc_bin.dump_bin(obj_file)

                if is_debug:
                    cb = _parse_cbuffer(root, asm_file)
                    if shader_type == 'ps':
                        cbuffers[entry_point] = cb
                    cbuffer_filename = (out_root + '.cbuffers.hpp').lower()
                    _save_cbuffer(cbuffer_filename, cb)

                if shader_file in LAST_FAIL_TIME:
                    del(LAST_FAIL_TIME[shader_file])

    if len(compile_res['fail']) == 0:
        _save_manifest(root, cbuffers, cbuffer_meta)
    return compile_res

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
        _safe_mkdir(OUT_DIR)
        cur_files = set()
        first_tick = True
        for full_path in glob.glob(os.path.join(SHADER_DIR, '*.hlsl')):
            _, filename = os.path.split(full_path)
            root, ext = os.path.splitext(filename)
            cur_files.add(root)
            if _should_compile(full_path):
                # If this is the first file we're processing, print the header
                if first_tick:
                    first_tick = False
                    ll = time.localtime()
                    print('==> COMPILE STARTED AT [%.2d:%.2d:%.2d]' % (
                        ll.tm_hour, ll.tm_min, ll.tm_sec))
                # the hlsl file has changed, so reparse its deps and entry
                # points
                entry_points, deps, cbuffer_meta = _parse_hlsl_file(full_path)
                meta_file = replace_ext(full_path, 'meta')
                if os.path.exists(meta_file):
                    _parse_meta_file(meta_file)
                SHADERS[root] = entry_points
                DEPS[root] = deps

                for row in open(full_path).readlines():
                    m = DEPS_RE.match(row)
                    if m:
                        dependant = m.groups()[0]
                        DEPS[root].add(dependant)

                _compile(full_path, root, cbuffer_meta)

        # purge data from files that no longer exist
        for root in prev_files.difference(cur_files):
            if root in SHADERS:
                del SHADERS[root]
            if root in DEPS:
                del DEPS[root]

        prev_files = cur_files

        time.sleep(1)
except KeyboardInterrupt:
    print('Exiting')
    exit(1)
