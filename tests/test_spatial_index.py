import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segmentation.tiling import SpatialGridIndex


class TestSpatialGridIndex(unittest.TestCase):

    def setUp(self):
        self.index = SpatialGridIndex(cell_size=1536)

    def test_01_same_cell_bbox(self):
        # Two bboxes in top-left cell (0, 0)
        self.index.add(0, [100, 100, 200, 200])
        self.index.add(1, [500, 500, 100, 100])

        cands = self.index.query_candidates([150, 150, 50, 50])
        self.assertEqual(cands, [0, 1])

    def test_02_neighboring_cell_bbox(self):
        # Item 0 in cell (0, 0), Item 1 in cell (1, 0) [x=1600..1800]
        self.index.add(0, [100, 100, 200, 200])
        self.index.add(1, [1600, 100, 200, 200])

        # Query cell (0,0) without gap
        cands_no_gap = self.index.query_candidates([100, 100, 200, 200], max_gap=0)
        self.assertEqual(cands_no_gap, [0])

        # Query cell (0,0) with max_gap=200 reaching into cell (1,0)
        cands_with_gap = self.index.query_candidates([1450, 100, 100, 100], max_gap=200)
        self.assertIn(0, cands_with_gap)
        self.assertIn(1, cands_with_gap)

    def test_03_overlapping_bbox(self):
        self.index.add(0, [100, 100, 500, 500])
        self.index.add(1, [300, 300, 500, 500])

        cands = self.index.query_candidates([200, 200, 200, 200])
        self.assertEqual(cands, [0, 1])

    def test_04_distant_bbox(self):
        # Item 0 in cell (0,0), Item 1 in cell (5,5) [x=7680, y=7680]
        self.index.add(0, [100, 100, 200, 200])
        self.index.add(1, [7680, 7680, 200, 200])

        cands = self.index.query_candidates([100, 100, 200, 200])
        self.assertEqual(cands, [0])
        self.assertNotIn(1, cands)

    def test_05_bbox_spanning_multiple_cells(self):
        # Large bbox from x=100 to x=3200 (spans cell 0, 1, 2)
        self.index.add(99, [100, 100, 3100, 500])

        # Query cell 0 (x=500)
        cands_cell0 = self.index.query_candidates([500, 100, 50, 50])
        self.assertIn(99, cands_cell0)

        # Query cell 2 (x=3100)
        cands_cell2 = self.index.query_candidates([3100, 100, 50, 50])
        self.assertIn(99, cands_cell2)

    def test_06_exact_cell_boundary(self):
        # Bbox sitting exactly on x=1536 boundary
        self.index.add(5, [1536, 1536, 100, 100])

        cands = self.index.query_candidates([1530, 1530, 20, 20], max_gap=15)
        self.assertIn(5, cands)

    def test_07_duplicate_candidate_removal(self):
        # A large bbox spanning cells (0,0), (0,1), (1,0), (1,1)
        self.index.add(42, [100, 100, 2000, 2000])

        # Query querying multiple cells
        cands = self.index.query_candidates([100, 100, 2000, 2000])
        # Even though 42 is in multiple cells, it must appear exactly once
        self.assertEqual(cands.count(42), 1)

    def test_08_deterministic_ordering(self):
        self.index.add(10, [100, 100, 50, 50])
        self.index.add(2, [100, 100, 50, 50])
        self.index.add(7, [100, 100, 50, 50])

        cands = self.index.query_candidates([100, 100, 50, 50])
        self.assertEqual(cands, [2, 7, 10])


if __name__ == "__main__":
    unittest.main()
