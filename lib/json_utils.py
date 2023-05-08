def decimal_default(obj):
    if isinstance(obj, (float, complex)):
        return format(obj, '.3f')
    return str(obj)
