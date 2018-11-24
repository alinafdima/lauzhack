
class Receipt:

    def __init__(self, filename):
        self.filename = filename

        # Data fields
        self.items = []
        self.total = None
        self.date = None
        self.store = None

        self.img = None
        self.img_text = None

        self.conn_comp_labels = None
        self.conn_comp_num = None

        self.logo = None
        self.patches = None
