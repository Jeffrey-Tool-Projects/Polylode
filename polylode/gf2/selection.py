"""Batch root discovery from serialized parents, one load per CAB."""
from collections import defaultdict

from ..core.paths import write_json_atomic


def roots_only(catalog, entries, output):
    groups = defaultdict(list)
    for entry in entries:
        groups[entry['cab']].append(entry)
    roots, skipped, failures = [], [], []
    print(f'ROOT_SCAN start: {len(groups)} CAB files; {len(entries)} candidates', flush=True)
    for completed, (cab, candidates) in enumerate(sorted(groups.items()), 1):
        env = catalog.environment()
        try:
            file = env.find_file(cab)
            for entry in candidates:
                go = file.objects[entry['id']].read()
                transform = next((c.component.read() for c in go.m_Component
                                  if c.component.type.name == 'Transform'), None)
                if transform is None:
                    skipped.append({'entry': entry, 'reason': 'no 3D Transform'})
                    continue
                if transform.m_Father:
                    skipped.append({'entry': entry, 'reason': 'child of another GameObject'})
                else:
                    roots.append(entry)
        except (ValueError, TypeError, KeyError, IndexError, OSError, EOFError,
                NotImplementedError, RuntimeError, AttributeError) as exc:
            # Keep failed candidates in the conversion queue for explicit per-item errors.
            failures.append({'cab': cab, 'error': f'{type(exc).__name__}: {exc}'})
            known = {(e['cab'], e['id']) for e in roots}
            roots.extend(e for e in candidates if (e['cab'], e['id']) not in known)
        finally:
            del env
        if completed % 25 == 0 or completed == len(groups):
            print(f'ROOT_SCAN {completed}/{len(groups)}; roots {len(roots)}; '
                  f'skipped {len(skipped)}; CAB errors {len(failures)}', flush=True)
    roots.sort(key=lambda e: (e['name'], e['cab'], e['id']))
    write_json_atomic(output, {'roots': roots, 'skipped': skipped, 'errors': failures}, indent=1)
    print(f'ROOTS {len(roots)}; children {len(skipped)}; CAB errors {len(failures)}', flush=True)
    return roots
