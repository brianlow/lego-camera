class Part:
    def __init__(self, part_num, name):
        self.part_num = part_num
        self.name = name

    @classmethod
    def from_db_row(cls, row):
        return cls(
            part_num=row[0],
            name=row[1]
        )

    def __repr__(self):
        return f"Part('{self.part_num}', '{self.name}')>"
