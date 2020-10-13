#!/usr/bin/env python3

import argparse
import enum
import json
import os
import re
import subprocess
import tempfile

from collections.abc import Sequence, Mapping

import rectpack


OPENSCAD_TAGMSG = re.compile(r'(ECHO|WARNING|ERROR): (.+)')
LSCAD_MSG = re.compile(r'"\[laserscad\] ##(.+)##"')
OPENSCAD_STR_ESC = re.compile(r'(["\\])')


class LaserSCADOp(enum.IntEnum):
    nop = 0
    pack = 1
    preview = 2
    engrave = 3
    cut = 4


def xvec2(xv2):
    x, y = xv2.split('x')
    return float(x), float(y)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('operation', choices=('cut', 'engrave', 'preview'), help='Operation.')
    p.add_argument('module', help='OpenSCAD module.')
    p.add_argument('pagesize', type=xvec2, help='Page size (in mmxmm).')
    p.add_argument('-D', '--define-vars', metavar='json', type=json.loads, help='Extra variable definitions passed to OpenSCAD, in a single JSON object.')
    p.add_argument('-o', '--output', metavar='dir', default='./cam', help='Output to directory (default to "./cam/").')
    return p, p.parse_args()

def generate_defines(params):
    def _serialize(v):
        # undefined
        if v is None:
            return f'undef'
        # ranges
        elif isinstance(v, range):
            return f'[{v.start}:{v.step}:{v.stop-1}]'
        # strings
        elif isinstance(v, str):
            v = OPENSCAD_STR_ESC.sub(r'\\\1', v)
            return f'"{v}"'
        # booleans
        elif isinstance(v, bool):
            return f'{"true" if v else "false"}'
        # numbers
        elif isinstance(v, int) or isinstance(v, float):
            return f'{repr(v)}'
        # vectors
        elif isinstance(v, Sequence):
            return f'[{", ".join(_serialize(e) for e in v)}]'
        # key-value pairs
        elif isinstance(v, Mapping):
            return f'[{", ".join(_serialize(e) for e in v.items())}]'
        # unknown python type. Fallback to repr()
        else:
            v = OPENSCAD_STR_ESC.sub(r'\\\1', repr(v))
            return f'"{v}"'

    for k, v in params.items():
        serialized = _serialize(v)
        yield '-D'
        yield f'{k}={serialized}'

def profile_to_params(profile):
    return {
        '_lpart_translation_table': profile['translation'],
        '_lpart_visibility_table': profile['visibility'],
        '_lpart_total_pages': profile['pages'],
        '_lpart_page_dim': profile['page_dim'],
    }

def extract_bb(openscad_module, openscad_exe='openscad', define_vars=None):
    print('=> Collecting information from OpenSCAD...')
    openscad_module = os.path.abspath(openscad_module)
    params = {}
    if define_vars is not None:
        params.update(define_vars)
    params['_laserscad_mode'] = int(LaserSCADOp.pack)
    defines = generate_defines(params)
    parts = 0
    with tempfile.NamedTemporaryFile(suffix='.echo') as openscad_output:
        subprocess.run([openscad_exe,
                        *defines,
                        '-o', openscad_output.name,
                        openscad_module])
        print('=> Extracting bounding boxes for lparts...')
        for lines in openscad_output:
            m = OPENSCAD_TAGMSG.match(lines.rstrip().decode('utf-8'))
            seen_ids = set()
            if m is not None:
                tag = m.group(1)
                msg = m.group(2)
                if tag == 'ECHO':
                    m_lscad = LSCAD_MSG.match(msg)
                    if m_lscad is not None:
                        part = m_lscad.group(1)
                        part_id, w, h = part.split(',')
                        if part in seen_ids:
                            print(f'** Duplicated lpart {repr(part_id)}.')
                            raise RuntimeError(f'Duplicated lpart {repr(part_id)}.')
                        seen_ids.add(part)
                        parts += 1
                        yield part_id, (float(w), float(h))
                elif tag == 'WARNING':
                    print(f'!! OpenSCAD: {msg}')
                elif tag == 'ERROR':
                    print(f'** OpenSCAD: {msg}')
                    raise RuntimeError(f'OpenSCAD: {msg}')
        print(f'==> Found {parts} lparts')

def pack_pages(bb, page_dim):
    '''
    Pack all bounding boxes to pages of specified size, with courtyard width (keepout area of each part) set to crtyd_width.
    '''
    result = {
        'translation': {},
        'visibility': {},
        'pages': None,
        'page_dim': None,
    }
    print('=> Packing...')
    packer = rectpack.newPacker(rotation=True)
    # TODO add page constraints and probably multiple page dimension support
    packer.add_bin(page_dim[0], page_dim[1], count=float('inf'))
    bb_buffered = dict(bb)
    for part_id, dim in bb_buffered.items():
        w, h = dim
        packer.add_rect(w, h, rid=part_id)
    packer.pack()
    
    for pageno, abin in enumerate(packer):
        print(f'==> Page #{pageno}')
        for rect in abin:
            x, y, w, h, part_id = rect.x, rect.y, rect.width, rect.height, rect.rid
            ow, oh = bb_buffered[part_id]
            # Check for rotation
            result['translation'][part_id] = (
                (x, y),
                ((0, 0, 90) if ow == h and oh == w else (0, 0, 0))
            )
            print(f'===> Part {repr(part_id)}: rot {result["translation"][part_id][1][2]} offset {result["translation"][part_id][0]}')
            result['visibility'][part_id] = pageno
    if len(result['translation']) != len(bb_buffered):
        print('!! Some parts are missing. Page size might be too small.')
    result['pages'] = len(packer)
    result['page_dim'] = page_dim
    return result

def export_cut_layer(profile, openscad_module, output_dir, openscad_exe='openscad', define_vars=None):
    print('=> Creating output directory...')
    openscad_module = os.path.abspath(openscad_module)
    os.makedirs(output_dir, exist_ok=True)
    output_dir_abs = os.path.abspath(output_dir)
    openscad_module_name = '.'.join(tuple(os.path.basename(openscad_module).split('.'))[:-1])

    for pageno in range(profile['pages']):
        output_path = os.path.join(output_dir_abs, f'{openscad_module_name}_{pageno}.dxf')
        print(f'=> Exporting page {pageno} to DXF {output_path}...')
        params = {}
        if define_vars is not None:
            params.update(define_vars)
        params.update(profile_to_params(profile))

        params['_lpart_current_page'] = pageno
        params['_laserscad_mode'] = int(LaserSCADOp.cut)

        defines = generate_defines(params)
        subprocess.run([openscad_exe,
                        *defines,
                        '-o', output_path,
                        openscad_module])

def start_preview(profile, openscad_module, openscad_exe='openscad', define_vars=None):
    print('=> Starting preview...')
    openscad_module = os.path.abspath(openscad_module)
    params = {}
    if define_vars is not None:
        params.update(define_vars)
    params.update(profile_to_params(profile))
    params['_laserscad_mode'] = int(LaserSCADOp.preview)
    defines = generate_defines(params)
    subprocess.run([openscad_exe,
                    *defines,
                    openscad_module])

if __name__ == '__main__':
    p, args = parse_args()

    bb = extract_bb(args.module, define_vars=args.define_vars)
    profile = pack_pages(bb, args.pagesize)

    if len(profile['translation']) == 0:
        print('!! No page generated. LaserSCAD may ran into some issues.')
    if args.operation == 'cut':
        export_cut_layer(profile, args.module, args.output, define_vars=args.define_vars)
    elif args.operation == 'preview':
        start_preview(profile, args.module, define_vars=args.define_vars)
