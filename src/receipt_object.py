
class Receipt:

    def __init__(self, filename):
        self.filename = filename

        # Data fields
        self.items = []
        self.total = None

        self.img = None

        self.conn_comp_labels = None
        self.conn_comp_num = None

        self.logo = None
        self.patches = None