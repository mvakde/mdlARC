"""Small, dependency-free parsers shared by the local Colab monitor."""

import math
import re


FIELDS = ("loss", "mtp_loss", "total_loss", "lr", "color_accuracy",
          "special_token_accuracy", "exact_output_accuracy")
VALUES = re.compile(r"\b([a-z_]+)=([-+\d.eE]+)%?")
EPOCH = re.compile(r"Epoch (\d+)/(\d+)")
STAGE = re.compile(r"Starting (training stage|evaluation) '([^']+)'")
BATCH = re.compile(r"\[([^]]+)\] (?:Finished|Accuracy) batch (\d+)\s*/\s*(\d+)")
ERROR = re.compile(r"^(?:Traceback \(|[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):)")


def parse_console(text):
    """Keep step losses separate from completed-epoch accuracies.

    A console tail may start halfway through a stage. A later stage marker
    resets the progress domain, so finetuning/evaluation cannot inflate speed.
    Repeated lines from recovered consoles overwrite observations, not counts.
    """
    result = {"stage": None, "kind": "training", "epoch": None,
              "epoch_total": None, "step": None, "steps_per_epoch": None,
              "step_metrics": {}, "epoch_metrics": {}, "evaluation": None,
              "error_observed": False}
    last_step = None
    for line in text.splitlines():
        match = STAGE.search(line)
        if match:
            result.update(stage=match[2], kind="evaluation" if match[1] == "evaluation" else "training",
                          epoch=None, epoch_total=None, step=None, steps_per_epoch=None,
                          step_metrics={}, epoch_metrics={}, evaluation=None, error_observed=False)
            last_step = None
        match = EPOCH.search(line)
        if match:
            result.update(epoch=int(match[1]), epoch_total=int(match[2]), step=0,
                          step_metrics={}, kind="training")
        values = {}
        for name, value in VALUES.findall(line):
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isfinite(number):
                values[name] = number
        if "epoch" in values:
            epoch = int(values["epoch"])
            metrics = {k: values[k] for k in FIELDS if k in values}
            if "step" in values:
                step = int(values["step"])
                result.update(epoch=epoch, step=step, step_metrics=metrics)
                last_step = (epoch, step)
            elif metrics:
                result["epoch_metrics"] = {"epoch": epoch, **metrics}
                result.update(epoch=epoch, step=None)
                if last_step and last_step[0] == epoch:
                    result["steps_per_epoch"] = last_step[1]
        match = BATCH.search(line)
        if match:
            result.update(kind="evaluation", evaluation={"split": match[1],
                          "batch": int(match[2]), "total_batches": int(match[3])})
        if ERROR.search(line):
            result["error_observed"] = True
    return result


def progress_value(progress):
    """Return a progress domain and fractional completed units, or unknown."""
    if progress.get("kind") == "evaluation":
        item = progress.get("evaluation")
        if item:
            return ("batches", progress.get("stage"), item["split"], item["total_batches"]), item["batch"]
        return None, None
    epoch = progress.get("epoch")
    count = progress.get("steps_per_epoch")
    if not epoch or not count:
        return None, None
    step = progress.get("step")
    value = float(epoch) if step is None else epoch - 1 + step / count
    return ("epochs", progress.get("stage"), progress.get("epoch_total"), count), value


def measure_speed(first, last, elapsed):
    left, start = progress_value(first)
    right, finish = progress_value(last)
    if left is None or right is None:
        return {"note": "Speed needs a completed epoch or an evaluation batch counter."}
    if left != right or finish < start:
        return {"note": "The stage or epoch length changed; take a new sample."}
    if elapsed < 5:
        return {"note": "Allow at least 5 seconds between samples."}
    delta = finish - start
    rate = delta / elapsed
    return {"unit": left[0], "per_second": rate,
            "seconds_per_unit": elapsed / delta if delta else None,
            "elapsed_seconds": elapsed, "units_advanced": delta,
            "note": "Estimate from logged progress; compilation, IO or pauses are included in this window."}


def compact_score(value):
    return {k: value[k] for k in ("score", "max_score", "percentage", "group", "timestamp") if k in value}
