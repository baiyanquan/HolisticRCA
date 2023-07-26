import json
import bisect


class TimeInterval:
    @staticmethod
    def generate_timestamp_list(fault_json_path, step_interval):
        with open(fault_json_path) as f:
            temp = json.load(f)
        timestamp_list = [int(temp['start']) + i * step_interval for i in
                          range(int((temp['end'] - temp['start']) / step_interval) + 1)]
        return timestamp_list

    @staticmethod
    def align_timestamp(timestamp_list, raw_timestamp):
        return bisect.bisect_left(timestamp_list, raw_timestamp)
