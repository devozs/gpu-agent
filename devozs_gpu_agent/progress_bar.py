"""tqdm-style progress bars for the model push (fetch-to-local).

We don't depend on tqdm (it isn't installed on every box), so this is a small,
faithful re-implementation of tqdm's default bar format:

    pytorch_model.bin 100%|████████████| 9.91M/9.91M [00:14<00:00, 690kB/s]

The exact same one-line string is used in two places:
  * the agent console — drawn live in-place (carriage-return) when stderr is a
    TTY, so an interactive run looks just like a tqdm download;
  * the management Logs panel — the *final* line per file is POSTed via the log
    endpoint, so the web UI shows the same bars (one per transferred file).

Sizes follow tqdm: SI units with a 1000 divisor ("9.91M", "28.9k") and no "B"
suffix; rates carry the unit ("690kB/s").
"""
import sys
import time

# Sub-cell block glyphs (1/8 increments), exactly tqdm's charset — lets the bar
# advance smoothly instead of jumping a whole cell at a time.
_BLOCKS = " ▏▎▍▌▋▊▉█"
DEFAULT_NCOLS = 24


def human_size(num, divisor=1000):
    """tqdm's format_sizeof: "9.91M", "28.9k", "690" — number + SI unit, no suffix."""
    num = float(num)
    for unit in ("", "k", "M", "G", "T", "P", "E", "Z"):
        if abs(num) < 999.5:
            if abs(num) < 99.95:
                if abs(num) < 9.995:
                    return f"{num:1.2f}{unit}"
                return f"{num:2.1f}{unit}"
            return f"{num:3.0f}{unit}"
        num /= divisor
    return f"{num:3.1f}Y"


def format_rate(bytes_per_sec):
    """A transfer rate in tqdm's style: "690kB/s", "2.03MB/s"."""
    return f"{human_size(bytes_per_sec)}B/s"


def format_interval(seconds):
    """tqdm's format_interval: "MM:SS" (or "H:MM:SS" past an hour)."""
    seconds = int(seconds)
    mins, s = divmod(seconds, 60)
    h, m = divmod(mins, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def render_bar(desc, n, total, elapsed, ncols=DEFAULT_NCOLS):
    """Render one tqdm-style line for `n`/`total` bytes after `elapsed` seconds."""
    total = max(int(total), 0)
    n = max(int(n), 0)
    if total:
        n = min(n, total)
    frac = (n / total) if total else 0.0
    if total:
        whole, rem = divmod(int(frac * ncols * 8), 8)
        bar = "█" * whole
        if whole < ncols:
            bar += _BLOCKS[rem] + " " * (ncols - whole - 1)
    else:
        bar = " " * ncols
    rate = (n / elapsed) if elapsed > 0 else 0
    remaining = ((total - n) / rate) if (rate > 0 and total) else 0
    stats = (f"{human_size(n)}/{human_size(total)} "
             f"[{format_interval(elapsed)}<{format_interval(remaining)}, {format_rate(rate)}]")
    prefix = f"{desc} " if desc else ""
    return f"{prefix}{frac * 100:3.0f}%|{bar}| {stats}"


class ConsoleBar:
    """A single live progress bar for one file's upload.

    On a TTY the bar is redrawn in place (throttled) as bytes flow; under
    journald/systemd (not a TTY) the intermediate redraws are skipped — only the
    final line is emitted, so the log doesn't fill with carriage-return spam.
    `close()` returns the finished line so the caller can also ship it to the UI.
    """

    def __init__(self, desc, total, stream=None, min_interval=0.1, ncols=DEFAULT_NCOLS):
        self._desc = desc
        self._total = max(int(total), 0)
        self._stream = stream if stream is not None else sys.stderr
        self._tty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._min_interval = min_interval
        self._ncols = ncols
        self._start = time.time()
        self._last_draw = 0.0
        self._n = 0

    @property
    def n(self):
        """Bytes sent so far — used to finalize a partial bar after a failed attempt."""
        return self._n

    def update(self, n, total=None):
        """Progress callback: `n` bytes sent so far (matches upload_model_file's reader)."""
        self._n = n
        if total is not None:
            self._total = max(int(total), 0)
        if not self._tty:
            return
        now = time.time()
        if n < self._total and (now - self._last_draw) < self._min_interval:
            return
        self._last_draw = now
        self._stream.write("\r" + render_bar(self._desc, n, self._total, now - self._start, self._ncols))
        self._stream.flush()

    def close(self, n=None):
        """Finalize the bar (drawn at `n`, default the full total) and return its text."""
        final = self._total if (n is None and self._total) else (self._n if n is None else n)
        line = render_bar(self._desc, final, self._total, time.time() - self._start, self._ncols)
        self._stream.write(("\r" if self._tty else "") + line + "\n")
        self._stream.flush()
        return line
