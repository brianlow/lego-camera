import sqlite3
from sqlite3 import Error
from lib.part import Part

DATABASE = "./bricks.db"

class Db:
    def __init__(self, g):
        self.g = g

    def get_conn(self):
        conn = getattr(self.g, '_conn', None)
        if conn is None:
            conn = self.g._database = sqlite3.connect(DATABASE)
        return conn

    def get_part_by_num(self, num):
        c = self.get_conn().cursor()
        c.execute("SELECT part_num, name FROM parts WHERE part_num=?", (num,))
        row = c.fetchone()
        if row is None:
            return None
        return Part.from_db_row(row)

    def get_ldraw_id_for_part_num(self, num):
        c = self.get_conn().cursor()
        sql = """
            select e.external_id
            from canonical_parts c
            left join external_ids e on e.part_num = c.part_num
            where c.canonical_part_num = ?
            and e."type" = 'LDraw'
            limit 1;
        """
        c.execute(sql, (num,))
        row = c.fetchone()
        if row is None:
            return None
        return row[0]

    def close(self):
        conn = getattr(self.g, '_database', None)
        if conn is None:
            return
        conn.close()
