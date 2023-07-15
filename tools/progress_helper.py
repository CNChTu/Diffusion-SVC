from rich.progress import Progress
from rich.progress import TextColumn, TaskProgressColumn, TimeRemainingColumn, BarColumn, ProgressColumn
from rich.style import StyleType
from rich.console import Console
from rich.logging import RichHandler
from rich.highlighter import NullHighlighter

from loguru import logger
import logging

from typing import List, Optional, Callable

logger.remove()
logger.add(RichHandler(highlighter=NullHighlighter(), show_level=False, show_time=False, show_path=False), colorize=True, diagnose=True, enqueue=True)

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

class TQDMLikeProgress:
    def __init__(
        self, total: float, description: str = "Working...",
        auto_refresh: bool = True,
        console: Optional[Console] = None,
        transient: bool = False,
        get_time: Optional[Callable[[], float]] = None,
        refresh_per_second: float = 10,
        style: StyleType = "bar.back",
        complete_style: StyleType = "bar.complete",
        finished_style: StyleType = "bar.finished",
        pulse_style: StyleType = "bar.pulse",
        update_period: float = 0.1,
        disable: bool = False,
        show_speed: bool = True):
        
        columns: List["ProgressColumn"] = (
            [TextColumn("[progress.description]{task.description}")] if description else []
        )
        columns.extend(
            (
                BarColumn(
                    style=style,
                    complete_style=complete_style,
                    finished_style=finished_style,
                    pulse_style=pulse_style,
                ),
                TaskProgressColumn(show_speed=show_speed),
                TimeRemainingColumn(elapsed_when_finished=True),
            )
        )
        self.progress = Progress(
            *columns,
            auto_refresh=auto_refresh,
            console=console,
            transient=transient,
            get_time=get_time,
            refresh_per_second=refresh_per_second or 10,
            disable=disable,
        )

        self.task_id = self.progress.add_task(description=description, total=total)
    
    def update(self, advance: Optional[float] = 1):
        self.progress.advance(self.task_id, advance)

    def close(self):
        self.progress.stop()
