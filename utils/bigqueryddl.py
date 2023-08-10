class Bigqueryddl:
    """
    Bigqueryddl class loads DDL files for various tables in a database.

    Attributes:
        ddl_dict (dict): dictionary of DDL files for various tables in a database.

    Methods:
        load_ddls: loads DDL files for various tables in a database.
    """

    def __init__(self):
        self.ddl_dict = self.load_ddls()

    @staticmethod
    def load_ddls():
        ddl_files = {
            "VIDEO_CONSULTS": "sql/ddl_video_consults.sql",
            "MARKET_PLACE": "sql/ddl_marketplace.sql",
            "PLATO_INVOICE": "sql/ddl_plato_invoice.sql",
            "USERS": "sql/ddl_users.sql",
        }

        ddl_dict = {}
        for table_name, file_name in ddl_files.items():
            with open(file_name, "r") as f:
                ddl_dict[table_name] = f.read()
        return ddl_dict
