import unittest

from anomalydetection.Arima.arima_case_detector import detect_case_kpis


class PacketLossExtractionTest(unittest.TestCase):
    def extract(self, kpi):
        raw = {"alarm_content": {"firstoccurutc": 1700000000}, "kpi": kpi}
        return detect_case_kpis({"raw_data": raw}, parameter_dir=None)[0]

    def test_peak_survives_segment_selection_and_units_agree(self):
        data = self.extract({
            "a": {"offline_loss_rate": {"unit": "%", "v": [
                [1699999500, 100], [1700000000, 0.5], [1700000060, 5]]}},
            "b": {"offline_loss_rate": {"unit": "ratio", "v": [
                [1699999400, 1], [1700000060, 0.15], [1700000120, 2]]}},
            "traffic_in": {"unit": "%", "v": [[1700000000, 83]]},
        })
        loss = data["offline_loss_rate"]
        self.assertEqual(loss["unit"], "ratio")
        self.assertEqual(loss["max_loss_rate"], 1)
        self.assertEqual(loss["max_loss_rate_time"], 1699999400)
        self.assertEqual(loss["upper_bound"], 0.05)
        self.assertIn(0.005, loss["abnormal_timing_segments"][0]["value"])
        self.assertEqual(data["traffic_in"]["abnormal_timing_segments"][0]["value"], [83])

    def test_fractional_threshold_and_no_segment(self):
        data = self.extract({
            "drop_packet_rate": {"unit": "fraction", "upper_bound": 0.02,
                                 "v": [[1699999300, 0.22], [1699999400, -1]]},
            "offline_loss_rate": {"unit": "%", "v": [[1700000000, 101]]},
        })
        self.assertNotIn("offline_loss_rate", data)
        self.assertEqual(data["drop_packet_rate"], {
            "unit": "ratio", "upper_bound": 0.02, "abnormal_timing_segments": [],
            "max_loss_rate": 0.22, "max_loss_rate_time": 1699999300,
        })
