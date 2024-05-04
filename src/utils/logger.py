from enum import Enum
from datetime import datetime
import json
from typing import List, Dict, Union, Tuple, Optional
import rich
from rich.console import Console, Text


class LogLevel(Enum):

    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    @staticmethod
    def from_str(value: str) -> "LogLevel":
        if value == "DEBUG":
            return LogLevel.DEBUG
        elif value == "INFO":
            return LogLevel.INFO
        elif value == "WARNING":
            return LogLevel.WARNING
        elif value == "ERROR":
            return LogLevel.ERROR
        else:
            raise ValueError(f"Unknown log level: {value}")


UTF_EMOJI_MAP = {
    LogLevel.DEBUG: "🐞",
    LogLevel.INFO: "ℹ️",
    LogLevel.WARNING: "⚠️",
    LogLevel.ERROR: "☠️",
}


class Logger:
    def __init__(
        self,
        basename: str,
        log_level: LogLevel,
        add_timestamp: bool = False,
        console: Optional[Console] = None,
    ):
        self.basename = basename
        self.instance_log_level = log_level
        self.instance_log_level_int = log_level.value
        self.add_timestamp = add_timestamp
        self.console = Console(emoji=True) if console is None else console

    def log(self, log_level: LogLevel, message: Union[str, Dict[str, object]]):
        text = Text()
        text.append(f"{UTF_EMOJI_MAP[log_level]} ", style="bold")
        text.append(f"[{self.basename}] ", style="yellow")
        if self.add_timestamp:
            text.append(f"[{datetime.now().strftime('%H:%M:%S')}] ", style="blue")
        if log_level.value >= self.instance_log_level_int:
            if isinstance(message, dict):
                text.append("\n" + json.dumps(message), style="green")
            else:
                text.append(message, style="violet")
            self.console.print(text)

    def debug(self, message: Union[str, Dict[str, object]]):
        self.log(LogLevel.DEBUG, message)

    def info(self, message: Union[str, Dict[str, object]]):
        self.log(LogLevel.INFO, message)

    def warning(self, message: Union[str, Dict[str, object]]):
        self.log(LogLevel.WARNING, message)

    def error(self, message: Union[str, Dict[str, object]]):
        self.log(LogLevel.ERROR, message)
