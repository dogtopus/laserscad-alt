#!/usr/bin/env python3

import argparse
import enum
import json
import math
import numbers
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

def open_module_in_openscad(module, openscad_exe='openscad', output=None, profile=None, ctx_vars=None, user_vars=None):
    openscad_module = os.path.abspath(module)
    params = {}
    if user_vars is not None:
        params.update(user_vars)
    if profile is not None:
        params.update(profile_to_params(profile))
    if ctx_vars is not None:
        params.update(ctx_vars)
    defines = generate_defines(params)
    openscad_args = [openscad_exe, *defines, openscad_module]
    if output is not None:
        openscad_args.append('-o')
        openscad_args.append(output)
    return subprocess.run(openscad_args)

def generate_defines(params):
    def _serialize(v):
        # undefined
        if v is None:
            return f'undef'
        # ranges
        elif isinstance(v, range):
            return f'[{v.start}:{v.step}:{v.stop-1}]'
        # strings (must be checked before Sequence)
        elif isinstance(v, str):
            v = OPENSCAD_STR_ESC.sub(r'\\\1', v)
            return f'"{v}"'
        # booleans (must be checked before numbers.Real because bool is also real)
        elif isinstance(v, bool):
            return f'{"true" if v else "false"}'
        # numbers
        elif isinstance(v, numbers.Real):
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
    parts = 0
    with tempfile.NamedTemporaryFile(suffix='.echo') as openscad_output:
        open_module_in_openscad(openscad_module,
                                openscad_exe=openscad_exe,
                                output=openscad_output.name,
                                ctx_vars={'_laserscad_mode': LaserSCADOp.pack},
                                user_vars=define_vars)
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
                    else:
                        print(f'==> OpenSCAD: {msg}')
                elif tag == 'WARNING':
                    print(f'!!! OpenSCAD: {msg}')
                elif tag == 'ERROR':
                    print(f'*** OpenSCAD: {msg}')
                    raise RuntimeError(f'OpenSCAD: {msg}')
        print(f'=> Found {parts} lparts')

def pack_pages(bb, page_dim):
    '''
    Pack all bounding boxes to pages of specified size.
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
    packer.add_bin(page_dim[0], page_dim[1], count=math.inf)
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

def export_layer(profile, openscad_module, output_dir, openscad_exe='openscad', define_vars=None, engrave=False):
    print('=> Creating output directory...')
    os.makedirs(output_dir, exist_ok=True)
    output_dir_abs = os.path.abspath(output_dir)
    openscad_module_name = '.'.join(tuple(os.path.basename(openscad_module).split('.'))[:-1])
    output_type = 'engrave' if engrave else 'cut'
    suffix = 'svg' if engrave else 'dxf'
    op = LaserSCADOp.engrave if engrave else LaserSCADOp.cut
    for pageno in range(profile['pages']):
        output_file = f'{openscad_module_name}_{output_type}_{pageno}.{suffix}'
        output_path = os.path.join(output_dir_abs, output_file)
        output_path_disp = os.path.join(output_dir, output_file)
        print(f'=> Exporting page {pageno} to {output_path_disp}...')
        open_module_in_openscad(openscad_module,
                                openscad_exe=openscad_exe,
                                output=output_path,
                                profile=profile,
                                ctx_vars={
                                    '_laserscad_mode': op,
                                    '_lpart_current_page': pageno,
                                },
                                user_vars=define_vars)
        if engrave:
            # TODO correct SVG offset.
            pass

def start_preview(profile, openscad_module, openscad_exe='openscad', define_vars=None):
    print('=> Starting preview...')
    open_module_in_openscad(openscad_module,
                            openscad_exe=openscad_exe,
                            profile=profile,
                            ctx_vars={'_laserscad_mode': LaserSCADOp.preview},
                            user_vars=define_vars)

if __name__ == '__main__':
    p, args = parse_args()

    bb = extract_bb(args.module, define_vars=args.define_vars)
    profile = pack_pages(bb, args.pagesize)

    if len(profile['translation']) == 0:
        print('!! No lpart was generated. LaserSCAD likely ran into a problem. See previous logs for more details.')
    if args.operation == 'cut':
        export_layer(profile, args.module, args.output, define_vars=args.define_vars)
    elif args.operation == 'engrave':
        export_layer(profile, args.module, args.output, define_vars=args.define_vars, engrave=True)
    elif args.operation == 'preview':
        start_preview(profile, args.module, define_vars=args.define_vars)
