"""
Functions for network matched-filter detection of seismic data.

Designed to cross-correlate templates generated by template_gen function
with data and output the detections.

:copyright:
    EQcorrscan developers.

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import ast
import copy
import os
import logging

import numpy as np
from obspy import Catalog, UTCDateTime, Stream
from obspy.core.event import (
    Comment, WaveformStreamID, Event, Pick, CreationInfo, ResourceIdentifier,
    Origin)

from eqcorrscan.core.match_filter.helpers import _test_event_similarity

Logger = logging.getLogger(__name__)


class Detection(object):
    """
    Single detection from detection routines in eqcorrscan.
    Information required for a full detection based on cross-channel \
    correlation sums.

    :type template_name: str
    :param template_name:
        The name of the template for which this detection was made.
    :type detect_time: obspy.core.utcdatetime.UTCDateTime
    :param detect_time: Time of detection as an obspy UTCDateTime object
    :type no_chans: int
    :param no_chans:
        The number of channels for which the cross-channel correlation sum
        was calculated over.
    :type detect_val: float
    :param detect_val:
        The raw value of the cross-channel correlation sum for this detection.
    :type threshold: float
    :param threshold:
        The value of the threshold used for this detection, will be the raw
        threshold value related to the cccsum.
    :type typeofdet: str
    :param typeofdet: Type of detection, STA, corr, bright
    :type threshold_type: str
    :param threshold_type: Type of threshold used for detection
    :type threshold_input: float
    :param threshold_input:
        Threshold set for detection, relates to `threshold` according to the
        `threshold_type`.
    :type chans: list
    :param chans: List of stations for the detection
    :type event: obspy.core.event.event.Event
    :param event:
        Obspy Event object for this detection, note that this is lost when
        writing to a :class:`Detection` objects to csv files using
        :func:`eqcorrscan.core.match_filter.Detection.write`
    :type id: str
    :param id: Identification for detection (should be unique).
    """

    def __init__(self, template_name, detect_time, no_chans, detect_val,
                 threshold, typeofdet, threshold_type, threshold_input,
                 chans=None, event=None, id=None):
        """Main class of Detection."""
        self.template_name = template_name
        self.detect_time = detect_time
        self.no_chans = int(no_chans)
        if not isinstance(chans, list):
            self.chans = [chans]
        else:
            self.chans = chans
        self.detect_val = np.float32(detect_val)
        self.threshold = np.float32(threshold)
        self.typeofdet = typeofdet
        self.threshold_type = threshold_type
        self.threshold_input = threshold_input
        self.event = event
        if id is not None:
            self.id = id
        else:
            self.id = (''.join(template_name.split(' ')) + '_' +
                       detect_time.strftime('%Y%m%d_%H%M%S%f'))
        if event is not None:
            event.resource_id = self.id
        if self.typeofdet == 'corr':
            assert round(abs(self.detect_val)) <= self.no_chans

    def __repr__(self):
        """Simple print."""
        print_str = ' '.join(
            ['template name =', self.template_name, '\n',
             'detection id =', self.id, '\n',
             'detection time =', str(self.detect_time), '\n',
             'number of channels =', str(self.no_chans), '\n',
             'channels =', str(self.chans), '\n',
             'detection value =', str(self.detect_val), '\n',
             'threshold =', str(self.threshold), '\n',
             'threshold type =', self.threshold_type, '\n',
             'input threshold =', str(self.threshold_input), '\n',
             'detection type =', str(self.typeofdet)])
        return "Detection(" + print_str + ")"

    def __str__(self):
        """Full print."""
        return (
            "Detection on template: {0} at: {1} with {2} channels: {3}".format(
                self.template_name, self.detect_time, self.no_chans,
                self.chans))

    def __eq__(self, other, verbose=False):
        for key in self.__dict__.keys():
            self_is_event = isinstance(self.event, Event)
            other_is_event = isinstance(other.event, Event)
            if key == 'event':
                if self_is_event and other_is_event:
                    if not _test_event_similarity(
                            self.event, other.event, verbose=verbose):
                        return False
                elif self_is_event and not other_is_event:
                    return False
                elif not self_is_event and other_is_event:
                    return False
            elif self.__dict__[key] != other.__dict__[key]:
                return False
        return True

    def __lt__(self, other):
        if self.detect_time < other.detect_time:
            return True
        else:
            return False

    def __le__(self, other):
        if self.detect_time <= other.detect_time:
            return True
        else:
            return False

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __hash__(self):
        """
        Cannot hash Detection objects, they may change.
        :return: 0
        """
        return 0

    def __ne__(self, other):
        return not self.__eq__(other)

    def _print_str(self):
        return "{0}; {1}; {2}; {3}; {4}; {5}; {6}; {7}; {8}".format(
            self.template_name, self.detect_time, self.no_chans,
            self.chans, self.detect_val, self.threshold,
            self.threshold_type, self.threshold_input, self.typeofdet)

    def copy(self):
        """
        Returns a copy of the detection.

        :return: Copy of detection
        """
        return copy.deepcopy(self)

    def write(self, fname, append=True):
        """
        Write detection to csv formatted file.

        Will append if append==True and file exists

        :type fname: str
        :param fname: Full path to file to open and write to.
        :type append: bool
        :param append:
            Set to true to append to an existing file, if True and file doesn't
            exist, will create new file and warn.  If False will overwrite
            old files.
        """
        mode = 'w'
        if append and os.path.isfile(fname):
            mode = 'a'
        header = '; '.join([
            'Template name', 'Detection time (UTC)', 'Number of channels',
            'Channel list', 'Detection value', 'Threshold', 'Threshold type',
            'Input threshold', 'Detection type'])
        with open(fname, mode) as _f:
            if mode == "w":
                _f.write(header + '\n')  # Write a header for the file
            _f.write(self._print_str() + '\n')

    def _calculate_event(self, template=None, template_st=None,
                         estimate_origin=True, correct_prepick=True):
        """
        Calculate an event for this detection using a given template.

        :type template: Template
        :param template: The template that made this detection
        :type template_st: `obspy.core.stream.Stream`
        :param template_st:
            Template stream, used to calculate pick times, not needed if
            template is given.
        :type estimate_origin: bool
        :param estimate_origin:
            Whether to include an estimate of the origin based on the template
            origin.
        :type correct_prepick: bool
        :param correct_prepick:
            Whether to apply the prepick correction defined in the template.
            Only applicable if template is not None

        .. rubric:: Note
            Works in place on Detection - over-writes previous events.
            Corrects for prepick if template given.
        """
        if template is not None and template.name != self.template_name:
            Logger.info("Template names do not match: {0}: {1}".format(
                template.name, self.template_name))
            return
        # Detect time must be valid QuakeML uri within resource_id.
        # This will write a formatted string which is still
        # readable by UTCDateTime
        det_time = str(self.detect_time.strftime('%Y%m%dT%H%M%S.%f'))
        ev = Event(resource_id=ResourceIdentifier(
            id=self.template_name + '_' + det_time,
            prefix='smi:local'))
        ev.creation_info = CreationInfo(
            author='EQcorrscan', creation_time=UTCDateTime())
        ev.comments.append(
            Comment(text="Template: {0}".format(self.template_name)))
        ev.comments.append(
            Comment(text='threshold={0}'.format(self.threshold)))
        ev.comments.append(
            Comment(text='detect_val={0}'.format(self.detect_val)))
        if self.chans is not None:
            ev.comments.append(
                Comment(text='channels used: {0}'.format(
                    ' '.join([str(pair) for pair in self.chans]))))
        if template is not None:
            template_st = template.st
            if correct_prepick:
                template_prepick = template.prepick
            else:
                template_prepick = 0
            try:
                template_picks = template.event.picks
            except AttributeError:
                template_picks = []
        else:
            template_prepick = 0
            template_picks = []
        min_template_tm = min(
            [tr.stats.starttime for tr in template_st])
        for tr in template_st:
            if (tr.stats.station, tr.stats.channel) \
                    not in self.chans:
                continue
            elif tr.stats.__contains__("not_in_original"):
                continue
            elif np.all(np.isnan(tr.data)):
                continue  # The channel contains no data and was not used.
            else:
                pick_time = self.detect_time + (
                    tr.stats.starttime - min_template_tm)
                pick_time += template_prepick
                new_pick = Pick(
                    time=pick_time, waveform_id=WaveformStreamID(
                        network_code=tr.stats.network,
                        station_code=tr.stats.station,
                        channel_code=tr.stats.channel,
                        location_code=tr.stats.location))
                template_pick = [p for p in template_picks
                                 if p.waveform_id.get_seed_string() ==
                                 new_pick.waveform_id.get_seed_string()]
                if len(template_pick) == 0:
                    new_pick.phase_hint = None
                elif len(template_pick) == 1:
                    new_pick.phase_hint = template_pick[0].phase_hint
                else:
                    # Multiple picks for this trace in template
                    similar_traces = template_st.select(id=tr.id)
                    similar_traces.sort()
                    _index = similar_traces.traces.index(tr)
                    try:
                        new_pick.phase_hint = sorted(
                            template_pick,
                            key=lambda p: p.time)[_index].phase_hint
                    except IndexError:
                        Logger.error(f"No pick for trace: {tr.id}")
                ev.picks.append(new_pick)
        if estimate_origin and template is not None\
                and template.event is not None:
            try:
                template_origin = (template.event.preferred_origin() or
                                   template.event.origins[0])
            except IndexError:
                template_origin = None
            if template_origin:
                for pick in ev.picks:
                    comparison_pick = [
                        p for p in template.event.picks
                        if p.waveform_id.get_seed_string() ==
                        pick.waveform_id.get_seed_string()]
                    comparison_pick = [p for p in comparison_pick
                                       if p.phase_hint == pick.phase_hint]
                    if len(comparison_pick) > 0:
                        break
                else:
                    Logger.error("Could not compute relative origin: no picks")
                    self.event = ev
                    return
                origin_time = pick.time - (
                        comparison_pick[0].time - template_origin.time)
                # Calculate based on difference between pick and origin?
                _origin = Origin(ResourceIdentifier(
                    id="EQcorrscan/{0}_{1}".format(
                        self.template_name, det_time), prefix="smi:local"),
                    time=origin_time, evaluation_mode="automatic",
                    evaluation_status="preliminary",
                    creation_info=CreationInfo(
                        author='EQcorrscan', creation_time=UTCDateTime()),
                    comments=[Comment(
                        text="Origin automatically assigned based on template"
                             " origin: use with caution.")],
                    latitude=template_origin.latitude,
                    longitude=template_origin.longitude,
                    depth=template_origin.depth,
                    time_errors=template_origin.time_errors,
                    latitude_errors=template_origin.latitude_errors,
                    longitude_errors=template_origin.longitude_errors,
                    depth_errors=template_origin.depth_errors,
                    depth_type=template_origin.depth_type,
                    time_fixed=False,
                    epicenter_fixed=template_origin.epicenter_fixed,
                    reference_system_id=template_origin.reference_system_id,
                    method_id=template_origin.method_id,
                    earth_model_id=template_origin.earth_model_id,
                    origin_type=template_origin.origin_type,
                    origin_uncertainty=template_origin.origin_uncertainty,
                    region=template_origin.region)
                ev.origins = [_origin]
        self.event = ev
        return self

    def extract_stream(self, stream, length, prepick):
        """
        Extract a cut stream of a given length around the detection.

        :type stream: `obspy.core.stream.Stream`
        :param stream: Stream of data to cut from
        :type length: float
        :param length: Length of data to extract in seconds.
        :type prepick: float
        :param prepick:
            Length before the expected pick on each channel to start the cut
            stream in seconds.

        :rtype: `obspy.core.stream.Stream`
        """
        assert self.event, "Detection must have an event - use Detection._" \
                           "calculate_event()"
        cut_stream = Stream()
        valid_chans = {
            (tr.stats.station, tr.stats.channel)
            for tr in stream}.intersection(set(self.chans))
        for station, channel in valid_chans:
            _st = stream.select(station=station, channel=channel)
            pick = [
                p for p in self.event.picks
                if p.waveform_id.station_code == station and
                p.waveform_id.channel_code == channel]
            if len(pick) == 0:
                Logger.info("No pick for {0}.{1}".format(station, channel))
                continue
            elif len(pick) > 1:
                Logger.info(
                    "Multiple picks found for {0}.{1}, using earliest".format(
                        station, channel))
                pick.sort(key=lambda p: p.time)
            pick = pick[0]
            cut_start = pick.time - prepick
            cut_end = cut_start + length
            _st = _st.slice(starttime=cut_start, endtime=cut_end).copy()
            # Minimum length check
            for tr in _st:
                if abs((tr.stats.endtime - tr.stats.starttime) -
                       length) < tr.stats.delta:
                    cut_stream += tr
                else:
                    Logger.info(
                        "Insufficient data length for {0}".format(tr.id))
        return cut_stream


def write_detections(detections, fname, mode='a'):
    """
    Write a list of detections to a file.

    :type detections: List of Detection
    :param detections: List of detection objects to write
    :type fname: str
    :param fname: Filename to write to
    :type mode: str
    :param mode: 'a' for append, or 'w' for write (will overwrite old files)
    """
    assert mode in {"a", "w"}
    lines = []
    if mode == "w" or not os.path.isfile(fname):
        lines.append('; '.join([
            'Template name', 'Detection time (UTC)', 'Number of channels',
            'Channel list', 'Detection value', 'Threshold', 'Threshold type',
            'Input threshold', 'Detection type']))
    lines.extend([d._print_str() for d in detections])
    lines = "\n".join(lines)
    lines += "\n"
    with open(fname, mode=mode) as f:
        f.write(lines)


def read_detections(fname, encoding="UTF8"):
    """
    Read detections from a file to a list of Detection objects.

    :type fname: str
    :param fname:
        File to read from, must be a file written to by Detection.write.
    :type encoding: str
    :param encoding: Encoding used for detection file.

    :returns: list of :class:`eqcorrscan.core.match_filter.Detection`
    :rtype: list

    .. note::
        :class:`eqcorrscan.core.match_filter.detection.Detection`'s returned
        do not contain Detection.event
    """
    with open(fname, "rb") as _f:
        lines = _f.read().decode(encoding).splitlines()
    detections = []
    for index, line in enumerate(lines):
        if line.rstrip().split('; ')[0] == 'Template name':
            continue  # Skip any repeated headers
        detection = line.rstrip().split('; ')
        detection[1] = UTCDateTime(detection[1])
        detection[2] = int(float(detection[2]))
        detection[3] = ast.literal_eval(detection[3])
        detection[4] = float(detection[4])
        detection[5] = float(detection[5])
        if len(detection) < 9:
            detection.extend(['Unset', float('NaN')])
        else:
            detection[7] = float(detection[7])
        detections.append(Detection(
            template_name=detection[0], detect_time=detection[1],
            no_chans=detection[2], detect_val=detection[4],
            threshold=detection[5], threshold_type=detection[6],
            threshold_input=detection[7], typeofdet=detection[8],
            chans=detection[3]))
    return detections


def write_catalog(detections, fname, format="QUAKEML"):
    """Write events contained within detections to a catalog file.

    :type detections: list
    :param detections: list of eqcorrscan.core.match_filter.Detection
    :type fname: str
    :param fname: Name of the file to write to
    :type format: str
    :param format: File format to use, see obspy.core.event.Catalog.write \
        for supported formats.
    """
    catalog = get_catalog(detections)
    catalog.write(filename=fname, format=format)


def get_catalog(detections):
    """
    Generate an :class:`obspy.core.event.Catalog` from list of \
    :class:`Detection`'s.

    :type detections: list
    :param detections: list of :class:`eqcorrscan.core.match_filter.Detection`

    :returns: Catalog of detected events.
    :rtype: :class:`obspy.core.event.Catalog`

    .. warning::
        Will only work if the detections have an event associated with them.
        This will not be the case if detections have been written to csv
        format using :func:`eqcorrscan.core.match_filter.Detection.write`
        and read back in.
    """
    catalog = Catalog()
    for detection in detections:
        if detection.event:
            catalog.append(detection.event)
    return catalog


if __name__ == "__main__":
    import doctest

    doctest.testmod()
