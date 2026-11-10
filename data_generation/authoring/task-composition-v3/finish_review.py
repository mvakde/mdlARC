#!/usr/bin/env python3
"""Checkpoint a completed review before opening improved generator/spec deliverables."""
from pathlib import Path
import time
from v2_storage import copy_file, read, sha, write


def finish(folder):
    folder = Path(folder)
    review = folder / 'review.md'
    marker = folder / 'review-complete.json'
    if review.is_symlink() or not review.is_file() or not review.read_text().strip():
        raise ValueError('Write a complete, nonempty review.md first')
    if marker.exists():
        record = read(marker)
        if record['review_sha256'] != sha(review) or record['source_generator_sha256'] != sha(folder / 'generator_v1.py'):
            raise ValueError('Completed review/source changed; preserve the checkpoint')
    else:
        if any((folder / name).exists() for name in ('generator.py', 'task_spec.md', 'review_coverage.md')):
            raise ValueError('Improvement deliverables exist before the review is complete')
        record = {'review_sha256': sha(review), 'source_generator_sha256': sha(folder / 'generator_v1.py'),
                  'completed_at_ns': time.time_ns()}
        write(marker, record)
    # Idempotent if interrupted between checkpoint and copying the starting files.
    for source, target in (('generator_v1.py', 'generator.py'), ('source_spec.md', 'task_spec.md')):
        if not (folder / target).exists():
            copy_file(folder / source, folder / target)
    return record


if __name__ == '__main__':
    finish(Path(__file__).resolve().parent)
    print('Review checkpoint saved. Now improve generator.py and task_spec.md; preserve review.md.')
