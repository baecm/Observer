# src/logger.py
class Logger:
    _level = "log"
    _levels = ["none", "log", "debug"]

    @classmethod
    def set_level(cls, level: str):
        if level not in cls._levels:
            print(f"[Logger] Invalid level '{level}', defaulting to 'none'")
            cls._level = "none"
        else:
            cls._level = level
            print(f"[Logger] Log level set to '{cls._level}'")

    @classmethod
    def debug(cls, *args):
        if cls._level == "debug":
            print("[DEBUG]", *args)

    @classmethod
    def log(cls, *args):
        if cls._level in ("log", "debug"):
            print("[LOG]", *args)

    @classmethod
    def info(cls, *args):
        if cls._level in ("log", "debug"):
            print("[INFO]", *args)

    @classmethod
    def warn(cls, *args):
        if cls._level in ("log", "debug"):
            print("[WARN]", *args)

    @classmethod
    def error(cls, *args):
        print("[ERROR]", *args)
