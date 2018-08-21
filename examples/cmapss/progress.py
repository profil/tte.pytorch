import sys

class Progress(object):
    def __init__(self):
        self.width = 0

    def update(self, values, end=False):
        info = ''
        for k, v in values:
            if type(v) == float:
                info += f' - {k}: {v:8.4f}'
            elif type(v) == int:
                info += f' - {k}: {v:4d}'

        if end:
            info += '\n'
            self.width = 0
        else:
            info += '.' * (self.width - len(info))
            sys.stdout.write('\b' * self.width)
            self.width = len(info)

        sys.stdout.write(info)
        sys.stdout.flush()
