from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from prefetch_admission import Observation,admit


class PrefetchAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.candidate=Observation(1,1.0,('session',1),'rgb-hash','processor-hash')
        self.selected=Observation(2,2.1,('session',1),'rgb-hash','processor-hash')
    def check(self,candidate=None,selected=None,context=('session',1)):
        return admit(candidate or self.candidate,selected or self.selected,current_context=context,action_started_at=2)
    def test_older_work_can_serve_identical_new_observation_without_relabeling(self):
        self.assertTrue(self.check()[0])
        self.assertEqual(self.selected.frame_id,2)
        self.assertEqual(self.selected.received_at,2.1)
    def test_old_selected_frame_is_rejected_even_if_pixels_match(self):
        self.assertEqual(self.check(selected=replace(self.selected,received_at=1.9))[1],'selected_frame_precedes_action')
    def test_changed_pixels_cannot_be_reused(self):
        self.assertEqual(self.check(selected=replace(self.selected,pixels='new-rgb'))[1],'pixels_changed')
    def test_changed_context_rejects_both_sources(self):
        self.assertFalse(self.check(context=('session',2))[0])
        self.assertFalse(self.check(candidate=replace(self.candidate,context=('old-session',1)))[0])
    def test_processing_change_rejects_features(self):
        self.assertEqual(self.check(selected=replace(self.selected,processing='changed-size'))[1],'processing_changed')
    def test_clock_domains_cannot_be_mixed(self):
        self.assertEqual(self.check(candidate=replace(self.candidate,clock='process-relative'))[1],'clock_domain_changed')
    def test_nonfinite_timestamps_are_not_admitted(self):
        for value in (float('nan'),float('inf'),True):
            self.assertEqual(self.check(selected=replace(self.selected,received_at=value))[1],'invalid_timestamp')
    def test_no_future_candidate_can_supply_an_earlier_observation(self):
        self.assertEqual(self.check(candidate=replace(self.candidate,received_at=2.2))[1],'candidate_from_future')


if __name__=='__main__':unittest.main()
