#!/usr/bin/env python
# coding: utf-8

__version__ = '1.0'

import argparse
import logging
import os
import os.path as fs
import itertools
import collections
import json

class DetectionKind(object):
    """true positive"""
    TP = 0
    """duplicate"""
    DUP = 1
    """false positive"""
    FP = 2

def calc_fbeta(prec, rec, beta):
    t = (1.0 + beta ** 2) * prec * rec

    #prec and rec are zero
    if t == 0.0:
        return 0.0
    else:
        return t / ( ( beta ** 2) * prec + rec)


def calc_f2(prec, rec):
    """F2 measure weights recall higher than precision"""
    return calc_fbeta(prec, rec, 2)

def calc_fmeasure(prec, rec):
    return calc_fbeta(prec, rec, 1)

#measures

def fmeasure(detected_sources, sources_cnt):
    prec = precision(detected_sources, sources_cnt)
    rec = recall(detected_sources, sources_cnt)
    return calc_fmeasure(prec, rec)

def f2(detected_sources, sources_cnt):
    prec = precision(detected_sources, sources_cnt)
    rec = recall(detected_sources, sources_cnt)
    return calc_f2(prec, rec)

def recall(detected_sources, sources_cnt):
    if sources_cnt <= 0:
        return 0.0

    return sum(1.0 for src in detected_sources
               if src == DetectionKind.TP) / sources_cnt

def precision(detected_sources, _):
    if detected_sources:
        return sum(1.0 for src in detected_sources
                   if src != DetectionKind.FP) / len(detected_sources)
    else:
        return 0.0

def avg_precision(detected_sources, sources_cnt):
    avg_prec = 0.0
    for src_num, src in enumerate(detected_sources):
        if src == DetectionKind.TP:
            avg_prec += precision(detected_sources[:src_num + 1], sources_cnt)

    return avg_prec / sources_cnt

def rprecision(detected_sources, sources_cnt):
    return precision(detected_sources[:sources_cnt], sources_cnt)


def _calc_avg_measure(all_detected_sources, all_sources_cnt, measure):
    return sum(
        measure(srcs, all_sources_cnt[src_num])
        for src_num, srcs in enumerate(all_detected_sources))/len(all_sources_cnt)

def mean_avg_precision(all_detected_sources, all_sources_cnt):
    return _calc_avg_measure(all_detected_sources, all_sources_cnt, avg_precision)

def avg_rprecision(all_detected_sources, all_sources_cnt):
    return _calc_avg_measure(all_detected_sources, all_sources_cnt, rprecision)


def macro_avg_precision(all_detected_sources, all_sources_cnt):
    return _calc_avg_measure(all_detected_sources, all_sources_cnt, precision)

def macro_avg_recall(all_detected_sources, all_sources_cnt):
    return _calc_avg_measure(all_detected_sources, all_sources_cnt, recall)

def micro_avg_precision(all_detected_sources, all_sources_cnt):
    return precision(sum(all_detected_sources, []),
                     sum(all_sources_cnt))

def micro_avg_recall(all_detected_sources, all_sources_cnt):
    return recall(itertools.chain(*all_detected_sources),
                  sum(all_sources_cnt))



# tools
class MeasureTitles(object):
    MEAN_AVG_PREC = "Mean average precision"
    RPRECISION = "R-precision"
    FMEASURE = "F1"
    F2 = "F2"
    RECALL = "Recall"
    PRECISION = "Precision"


class BaseCalcOpts(object):
    def __init__(self, micro = False):
        self.micro = micro


class BaseCalc(object):
    def __init__(self, opts, detections_index, sources_index,
                 duplicates_tester = None):
        """Index is a dict:
        {"suspicious_id" : [
          { "id": 2328},
          { "id": 23}...
          ...
        ]
        """
        super(BaseCalc, self).__init__()
        self._opts              = opts
        self._detections_index  = collections.OrderedDict()
        self._sources_index     = collections.OrderedDict(sources_index.items())
        self._duplicates_tester = duplicates_tester
        self._transform_detections(detections_index)

    def _transform_detections(self, detections_index):
        for susp_id in self._sources_index:
            detections = detections_index.get(susp_id, [])
            try:
                self._try_transform_one_detection(susp_id, detections)
            except Exception as e:
                logging.error("Failed to parse meta for %s: %s", susp_id, e)

    def _is_dupl(self, susp_id, det):
        if det["id"] == susp_id:
            #the found document is the query document
            return True

        if self._duplicates_tester is not None:
            return self._duplicates_tester(susp_id, det)


    def _try_transform_one_detection(self, susp_id, detections):
        true_sources = frozenset(src["id"] for src in self._sources_index[susp_id])

        annotated_detections = []
        for det in detections:
            if det["id"] in true_sources:
                annotated_detections.append(DetectionKind.TP)
            elif self._is_dupl(susp_id, det):
                annotated_detections.append(DetectionKind.DUP)
            else:
                annotated_detections.append(DetectionKind.FP)

        self._detections_index[susp_id] = annotated_detections


    def _get_sources_cnt(self):
        return [len(self._sources_index[susp_id]) for susp_id in self._sources_index]


class Calc(BaseCalc):
    def __init__(self, opts, detections_index, sources_index,
                 duplicates_tester = None):
        super(Calc, self).__init__(opts, detections_index, sources_index,
                                   duplicates_tester)

    def __call__(self):
        all_sources_cnt = self._get_sources_cnt()
        if self._opts.micro:
            prec = micro_avg_precision(self._detections_index.itervalues(),
                                       all_sources_cnt)
            rec = micro_avg_recall(self._detections_index.itervalues(),
                                   all_sources_cnt)

        else:
            prec = macro_avg_precision(self._detections_index.itervalues(),
                                       all_sources_cnt)
            rec = macro_avg_recall(self._detections_index.itervalues(),
                                   all_sources_cnt)

        mean_avg_prec = mean_avg_precision(self._detections_index.itervalues(),
                                           all_sources_cnt)

        rprec = avg_rprecision(self._detections_index.itervalues(),
                               all_sources_cnt)

        return collections.OrderedDict([
            (MeasureTitles.MEAN_AVG_PREC, mean_avg_prec),
            (MeasureTitles.RPRECISION, rprec),
            (MeasureTitles.FMEASURE, calc_fmeasure(prec, rec)),
            (MeasureTitles.F2, calc_f2(prec, rec)),
            (MeasureTitles.RECALL, rec),
            (MeasureTitles.PRECISION, prec)
        ])


def load_sources(path, sources_key = "plagiarism"):
    with open(path, 'r') as f:
        json_obj = json.load(f)
        susp_file = json_obj["suspicious-document"]
        if susp_file.endswith(".txt"):
            susp_id = susp_file[:-4]
        else:
            susp_id = susp_file

    return susp_id, json_obj[sources_key]


def load_sources_from_dir(dir_path, sources_key = "plagiarism"):
    sources_index = {}
    entries = [e for e in os.listdir(dir_path)
               if e.endswith(".json")]
    for entry in entries:
        file_path= fs.join(dir_path, entry)
        try:
            susp_id, sources = load_sources(file_path, sources_key)
            sources_index[susp_id] = sources
        except Exception as excep:
            logging.error("Failed to load sources from file %s: %s", file_path, excep)
    return sources_index




def run(opts):
    print 'Reading', opts.plag_path
    plag_sources = load_sources_from_dir(opts.plag_path)
    print 'Reading', opts.det_path
    detected_sources = load_sources_from_dir(opts.det_path, "detected-plagiarism")

    display_measures = [MeasureTitles.MEAN_AVG_PREC, MeasureTitles.FMEASURE,
                        MeasureTitles.RECALL, MeasureTitles.PRECISION]
    measures = Calc(opts, detected_sources, plag_sources)()
    for measure in display_measures:
        print "%s %.3f" % (measure, measures[measure])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", default = False)

    parser.add_argument("--plag_path", "-p", required = True,
                        help = "Path to the json files with plagiarism sources")
    parser.add_argument("--det_path", "-d", required = True,
                        help = "Path to the json files with detected sources")
    parser.add_argument("--micro", "-m", action="store_true", default=False)
    args = parser.parse_args()

    FORMAT="%(asctime)s %(levelname)s: %(name)s: %(message)s"
    logging.basicConfig(level = logging.DEBUG if args.verbose else logging.INFO,
                        format = FORMAT)
    try:

        run(args)
    except Exception as e:
        logging.exception("Failed to run: %s ", e)


if __name__ == '__main__' :
    main()


# tests
import unittest

DK = DetectionKind
MT = MeasureTitles


class MAPTestCase(unittest.TestCase):
    def simple_test(self):
        detections = [
            DK.FP,
            DK.TP,
            DK.FP
        ]

        val = avg_precision(detections, 2)
        self.assertAlmostEqual(0.25, val, 3)

        detections.append(DK.TP)
        val = avg_precision(detections, 2)
        self.assertAlmostEqual(0.5, val, 3)

    def test_wit_dups(self):
        detections = [DK.TP, DK.DUP, DK.TP, DK.DUP]
        val = avg_precision(detections, 2)
        self.assertAlmostEqual(1.0, val, 3)


class RecallTestCase(unittest.TestCase):
    def simple_test(self):
        detections = [DK.TP, DK.DUP, DK.FP, DK.TP, DK.FP, DK.FP]
        val = recall(detections, 4)
        self.assertAlmostEqual(0.5, val, 3)

    def zero_found_test(self):
        detections = [DK.FP, DK.FP, DK.FP, DK.FP, DK.FP, DK.FP]
        val = recall(detections, 4)
        self.assertAlmostEqual(0.0, val, 3)


class PrecisionTestCase(unittest.TestCase):
    def simple_test(self):
        detections = [DK.TP, DK.DUP, DK.FP, DK.TP, DK.FP, DK.FP]
        val = precision(detections, 4)
        self.assertAlmostEqual(0.5, val, 3)

    def zero_found_test(self):
        detections = [DK.FP, DK.FP, DK.FP, DK.FP, DK.FP, DK.FP]
        val = precision(detections, 4)
        self.assertAlmostEqual(0.0, val, 3)


class MicroTestCase(unittest.TestCase):
    def simple_prec_test(self):
        det1 = [DK.TP, DK.DUP, DK.FP, DK.TP, DK.FP, DK.FP]
        det2 = [DK.TP, DK.DUP, DK.TP]

        src_cnts = [4, 2]

        val = micro_avg_precision([det1, det2], src_cnts)

        self.assertAlmostEqual(6/9.0, val, 3)

    def simple_rec_test(self):
        det1 = [DK.TP, DK.DUP, DK.FP, DK.TP, DK.FP, DK.FP]
        det2 = [DK.TP, DK.DUP]

        src_cnts = [4, 1]

        val = micro_avg_recall([det1, det2], src_cnts)
        self.assertAlmostEqual(0.6, val, 3)

class MacroTestCase(unittest.TestCase):
    def simple_prec_test(self):
        det1 = [DK.TP, DK.DUP, DK.FP, DK.TP, DK.FP, DK.FP] #0.5
        det2 = [DK.TP, DK.DUP, DK.TP] #1.0

        src_cnts = [4, 2]

        val = macro_avg_precision([det1, det2], src_cnts)

        self.assertAlmostEqual(0.75, val, 3)

    def simple_rec_test(self):
        det1 = [DK.TP, DK.DUP, DK.FP, DK.TP, DK.FP, DK.FP] #0.5
        det2 = [DK.TP, DK.DUP] #1.0

        src_cnts = [4, 1]

        val = macro_avg_recall([det1, det2], src_cnts)
        self.assertAlmostEqual(0.75, val, 3)

class CalcTestCase(unittest.TestCase):


    def _create_calc(self, detections, micro = False):
        sources_index = {
            "1" : [{"id": "10"}, {"id": "11"}, {"id": "12"}],
            "2" : [{"id": "20"}, {"id": "21"}],
            "3" : [{"id": "30"}, {"id": "31"}, {"id": "32"}, {"id": "33"}]}
        opts = BaseCalcOpts(micro)

        return Calc(opts, detections, sources_index)

    def simple_test(self):
        detections_index = {
            "1" : [{"id": "10"}, {"id": "11"}, {"id": "12"}],
            "2" : [{"id": "20"}, {"id": "21"}],
            "3" : [{"id": "30"}, {"id": "31"}, {"id": "32"}, {"id": "33"}]}
        calc = self._create_calc(detections_index)
        measures = calc()
        self.assertEqual(1.0, measures[MT.FMEASURE])
        self.assertEqual(1.0, measures[MT.MEAN_AVG_PREC])

    def detection_missing_test(self):
        detections_index = {
            "1" : [{"id": "10"}, {"id": "11"}, {"id": "12"}],
            "2" : [{"id": "20"}, {"id": "21"}]
            }
        calc = self._create_calc(detections_index)
        measures = calc()
        # print measures
        self.assertAlmostEqual(2/3.0, measures[MT.PRECISION])
        self.assertAlmostEqual(2/3.0, measures[MT.MEAN_AVG_PREC])

    def partial_detection_test(self):
        detections_index = {
            "1" : [{"id": "40"}, {"id": "41"}, {"id": "12"}, {"id": "42"}],
            "2" : [{"id": "20"}, {"id": "43"}],
            "3" : [{"id": "44"}, {"id": "32"}, {"id": "31"}, {"id": "45"}, {"id": "46"}, {"id": "47"}]
            }
        calc = self._create_calc(detections_index)
        measures = calc()

        #1; prec - 1/4, rec - 1/3, rprec - 1/3, ap - 1/9
        #2; prec - 1/2, rec - 1/2, rprec - 1/2, ap - 1/2
        #3; prec - 2/6, rec - 2/4, rprec - 2/4, ap - 7/24
        self.assertAlmostEqual(13/12.0/3.0, measures[MT.PRECISION])
        self.assertAlmostEqual(16/12.0/3.0, measures[MT.RECALL])
        self.assertAlmostEqual(0.398, measures[MT.FMEASURE], 3)
        self.assertAlmostEqual(16/12.0/3.0, measures[MT.RPRECISION])
        self.assertAlmostEqual(65/72.0/3.0, measures[MT.MEAN_AVG_PREC])

    def micro_test(self):
        detections_index = {
            "1" : [{"id": "40"}, {"id": "41"}, {"id": "12"}, {"id": "42"}],
            "2" : [{"id": "20"}, {"id": "43"}],
            "3" : [{"id": "44"}, {"id": "32"}, {"id": "31"}, {"id": "45"}, {"id": "46"}, {"id": "47"}]
            }
        calc = self._create_calc(detections_index, micro = True)
        measures = calc()

        self.assertAlmostEqual(4/12.0, measures[MT.PRECISION])
        self.assertAlmostEqual(4/9.0, measures[MT.RECALL])

    def dupl_test(self):
        detections_index = {
            "1" : [{"id": "1"}, {"id": "10"}, {"id": "11"}, {"id": "12"}],
            "2" : [{"id": "2"}, {"id": "20"}, {"id": "21"}],
            "3" : [{"id": "3"}, {"id": "30"}, {"id": "31"}, {"id": "32"}, {"id": "33"}]}

        calc = self._create_calc(detections_index)
        measures = calc()
        self.assertEqual(1.0, measures[MT.FMEASURE])
        self.assertEqual(1.0, measures[MT.MEAN_AVG_PREC])
