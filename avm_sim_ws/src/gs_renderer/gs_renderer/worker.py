"""One persistent render process, one queued snapshot, no global scene state."""
from __future__ import annotations
import multiprocessing as mp
from queue import Empty
import traceback
from .pipeline import SensorPipeline


def _work(requests, responses, backend: str) -> None:
    # The parent owns process shutdown. Ignoring SIGINT keeps Ctrl+C from
    # interrupting NumPy work before SensorWorker.close() can stop us cleanly.
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    pipeline = SensorPipeline(backend)
    while True:
        request = requests.get()
        if request is None:
            return
        revision,scene = request
        try:
            result = pipeline.render(scene)
            responses.put((revision,scene,result,""))
        except Exception:
            responses.put((revision,scene,{},traceback.format_exc()))


class SensorWorker:
    def __init__(self, backend: str):
        ctx = mp.get_context("spawn")
        self.requests,self.responses = ctx.Queue(maxsize=1),ctx.Queue(maxsize=1)
        self.process = ctx.Process(target=_work,args=(self.requests,self.responses,backend),daemon=True)
        self.process.start()
        self.busy = False

    def submit(self, revision: int, scene) -> None:
        if self.busy:
            raise RuntimeError("Only one render may be in flight")
        self.requests.put((revision,scene))
        self.busy = True

    def poll(self):
        if not self.process.is_alive():
            raise RuntimeError(f"Sensor worker exited with code {self.process.exitcode}")
        try:
            response = self.responses.get_nowait()
            self.busy = False
            return response
        except Empty:
            return None

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self.requests.put_nowait(None)
            except __import__("queue").Full:
                pass
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        self.requests.close()
        self.responses.close()
