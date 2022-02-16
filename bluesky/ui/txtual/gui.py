""" Textual Tui for BlueSky."""

from textual.app import App

import bluesky as bs
from bluesky.ui.txtual.tuiclient import TuiClient

def start(mode):

    # Start the bluesky network client
    client = TuiClient()

    # else:
    client.connect(event_port=bs.settings.event_port,
                       stream_port=bs.settings.stream_port)

    # Start the Qt main loop
    app.exec_()

from textual.app import App
from textual.widgets import Placeholder
from textual.widget import Widget
from textual.reactive import Reactive
from rich.panel import Panel
class Commandline(Widget):
    line = Reactive("")
    can_focus = True
    def on_key(self, event):
        if event.key in ('backspace', 'ctrl+h'):
            self.line = self.line[:-1]
        elif event.key == 'enter':
            self.line = ''
        else:
            self.line += event.key
    def render(self) -> Panel:
        return Panel("[b]>>[/b]" + self.line)

class SimpleApp(App):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cmdtext = ''
        self.cmdline = Commandline()
    async def on_mount(self) -> None:
        await self.view.dock(self.cmdline, edge="bottom", size=3)
        await self.view.dock(Placeholder(), edge="top")
        await self.set_focus(self.cmdline)

SimpleApp.run(log="textual.log")