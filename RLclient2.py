from PyQt6.QtWidgets import QTextEdit, QApplication
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer

from bluesky.network.client import Client


class TextClient(Client):
    '''
        Subclassed Client with a timer to periodically check for incoming data,
        an overridden event function to handle data, and a stack function to
        send stack commands to BlueSky.
    '''
    modes = ['Init', 'Hold', 'Operate', 'End']

    def __init__(self, actnode_topics=b''):
        super().__init__(actnode_topics)
        self.subscribe(b'SIMINFO')
        self.subscribe(b'ACDATA')
        self.timer = QTimer()
        self.timer.timeout.connect(self.receive)
        self.timer.start(20)

    def event(self, name, data, sender_id):
        ''' Overridden event function. '''
        print(name)

    def actnode_changed(self, newact):
        pass

    def stream(self, name, data, sender_id):
        if name == b'SIMINFO':
            speed, simdt, simt, simutc, ntraf, state, scenname = data
           
        if name == b'ACDATA':
            pass

class Echobox(QTextEdit):
    ''' Text box to show echoed text coming from BlueSky. '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setReadOnly(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def echo(self, text, flags=None):
        ''' Add text to this echo box. '''
        self.append(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class Cmdline(QTextEdit):
    ''' Wrapper class for the command line. '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(21)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def keyPressEvent(self, event):
        ''' Handle Enter keypress to send a command to BlueSky. '''
        if event.key() == Qt.Key.Key_Enter or event.key() == Qt.Key.Key_Return:
            if bsclient is not None:
                bsclient.stack(self.toPlainText())
                echobox.echo(self.toPlainText())
            self.setText('')
        else:
            super().keyPressEvent(event)


if __name__ == '__main__':
    # Construct the Qt main object
    app = QApplication([])

    # Create a window with a stack text box and a command line
    win = QWidget()
    win.setWindowTitle('Example external client for BlueSky')
    layout = QVBoxLayout()
    win.setLayout(layout)

    echobox = Echobox(win)
    cmdline = Cmdline(win)
    layout.addWidget(echobox)
    layout.addWidget(cmdline)
    win.show()

    # Create and start BlueSky client
    bsclient = TextClient()
    bsclient.connect(event_port=11000, stream_port=11001)

    # Start the Qt main loop
    app.exec()