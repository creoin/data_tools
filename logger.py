class Logger(object):
    def __init__(self, *quantities):
        self.logs = {}
        self.n_quantities = len(quantities)
        self.names = quantities

        for quantity in quantities:
            self.logs[quantity] = []

    def log(self, **quantities):
        assert len(quantities) == self.n_quantities
        for name, val in quantities.items():
            self.logs[name].append(val)

    def printlog(self):
        rows = self._get_rows()
        for name in self.names:
            print('{:20s}'.format(name), end='')
        print()
        for row in rows:
            for name in self.names:
                if name.lower() in ['iteration', 'step', 'gstep', 'global_step']:
                    print('{:<20}'.format(row[name]), end='')
                else:
                    print('{:<20.6f}'.format(row[name]), end='')
            print()

    def write_csv(self, filename):
        rows = self._get_rows()
        with open(filename,'w') as file:
            headerline = ','.join(self.names)
            file.write(headerline)
            file.write('\n')
            for row in rows:
                writelist = [str(row[name]) for name in self.names]
                writeline = ','.join(writelist)
                file.write(writeline)
                file.write('\n')

    def _get_rows(self):
        rows = [dict(zip(self.names, row)) for row in zip(*(self.logs[k] for k in self.names))]
        return rows
