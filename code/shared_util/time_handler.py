from datetime import datetime
import pytz


class TimeHandler:
    @staticmethod
    def datetime_to_timestamp(datetime_str: str) -> int:
        timezone = pytz.timezone("Asia/Shanghai")
        dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        return int(timezone.localize(dt).timestamp())

    @staticmethod
    def timestamp_to_datetime(timestamp: int):
        timezone = pytz.timezone("Asia/Shanghai")
        dt = pytz.datetime.datetime.fromtimestamp(timestamp, timezone)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
