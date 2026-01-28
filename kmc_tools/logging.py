from dataclasses import dataclass, asdict
import uuid
import os
import threading
import queue
from typing import List, Dict, Any
import json
from abc import ABC, abstractmethod


class LogParams:
    def __init__(self, data_dir: str, name: str) -> None:
        self.name = name
        self.data_dir = data_dir
        self.log_path = _log_init(self)
        self.logfile = os.path.join(self.log_path, f"log.{name}")


def _log_init(params: LogParams) -> str:
    """
    Makes a directory in the data dir, it is in the format '{name}-{hash} where hash is a unique uuid'
    """
    assert " " not in params.name, "Names must not contain spaces"
    hash = str(uuid.uuid4())[:8]
    dir: str = os.path.abspath(params.data_dir) + f"/{params.name}-{hash}"

    if os.path.isdir(dir):
        print("You need to buy a lottery ticket immediately")

    os.mkdir(dir)

    return dir


class LoggerThread(threading.Thread):
    def __init__(
        self,
        logparams: LogParams,
        logqueue: queue.Queue,
        flush_interval: float = 1,
        batch_size: float = 20,
    ) -> None:
        super().__init__(daemon=True)
        self.logparams: LogParams = logparams
        self._stop = threading.Event()
        self.log_queue = logqueue
        self.flush_interval = flush_interval
        self.batch_size = batch_size

    def run(self) -> None:
        buffer: List = []
        with open(self.logparams.logfile, "a") as f:
            while True:
                try:
                    msg = self.log_queue.get(timeout=self.flush_interval)
                except queue.Empty:
                    msg = None

                if msg is None:
                    if buffer:
                        f.write("".join(buffer))
                        buffer.clear()
                    if self._stop.is_set():
                        break
                    continue

                buffer.append(json.dumps(msg))

                if len(buffer) >= self.batch_size:
                    f.write("".join(buffer))
                    buffer.clear()


@dataclass(frozen=True)
class LogMessage:
    name: str
    type: str
    time: float
    payload: Dict[str, Any]


class LogEmitter(ABC):
    @abstractmethod
    def emit(self, msg: LogMessage): ...


class QueueEmitter(LogEmitter):
    def __init__(self, logqueue) -> None:
        self._logqueue: queue.Queue = logqueue

    def emit(self, msg: LogMessage):
        self._logqueue.put(asdict(msg))
