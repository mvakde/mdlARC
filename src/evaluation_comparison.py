"""Paired ARC scores and demonstration-count breakdowns from saved predictions."""
import json
from pathlib import Path


def compare_submissions(reference, current, solutions, challenges):
    def task_score(submission, task, truth):
        attempts = submission.get(task, [])
        return sum(i < len(attempts) and any(attempts[i].get(key) == grid
                   for key in ('attempt_1', 'attempt_2')) for i, grid in enumerate(truth)) / len(truth) if truth else 0.
    rows, groups = [], {}
    for task, truth in sorted(solutions.items()):
        old, new = task_score(reference, task, truth), task_score(current, task, truth)
        count = sum(p.get('output') is not None for p in challenges[task]['train'])
        rows.append(dict(task_id=task, demos=count, all_demo_score=old, loo_score=new, change=new-old))
        group = groups.setdefault(str(count), dict(tasks=0, all_demo_score=0., loo_score=0., change=0.))
        group['tasks'] += 1; group['all_demo_score'] += old; group['loo_score'] += new; group['change'] += new-old
    return dict(tasks=rows, by_demo_count=groups,
        all_demo_score=sum(r['all_demo_score'] for r in rows),
        loo_score=sum(r['loo_score'] for r in rows), change=sum(r['change'] for r in rows))


def write_comparison(reference_path, submission_path, solutions_path, challenges_path):
    paths = [reference_path, submission_path, solutions_path, challenges_path]
    report = compare_submissions(*(json.loads(Path(p).read_text()) for p in paths))
    output = Path(submission_path).parent / 'all_demo_comparison.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Paired all-demo comparison: {report['all_demo_score']} -> {report['loo_score']}; saved {output}")
