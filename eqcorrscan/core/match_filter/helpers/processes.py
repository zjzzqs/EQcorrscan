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
import os
import tempfile
import time
import traceback
import logging
import numpy as np

from typing import List, Union, Iterable
from timeit import default_timer

from multiprocessing import Queue
from queue import Empty

from obspy import Stream

from eqcorrscan.core.match_filter.helpers import (
    _pickle_stream, _unpickle_stream)
from eqcorrscan.core.match_filter.helpers.tribe import (
    _download_st, _pre_process, _group, _detect,
    _read_template_db, _make_party)

from eqcorrscan.utils.correlate import (
    _get_array_dicts, _fmf_stabilisation, _fmf_reshape)
from eqcorrscan.utils.pre_processing import (
    _quick_copy_stream, _prep_data_for_correlation)


Logger = logging.getLogger(__name__)


###############################################################################
#                           Process handlers
###############################################################################


class Poison(Exception):
    """
    Exception passing within EQcorrscan

    :type value: Exception
    :param value: Exception to pass between processes
    """
    def __init__(self, value):
        """
        Poison Exception.
        """
        self.value = value

    def __repr__(self):
        return f"Poison({self.value.__repr__()})"

    def __str__(self):
        """
        >>> print(Poison(Exception('alf')))
        Poison(Exception('alf'))
        """
        return self.__repr__()


def _get_and_check(input_queue: Queue, poison_queue: Queue, step: float = 0.5):
    """
    Get from a queue and check for poison - returns Poisoned if poisoned.

    :param input_queue: Queue to get something from
    :param poison_queue: Queue to check for poison

    :return: Item from queue or Poison.
    """
    while True:
        poison = _check_for_poison(poison_queue)
        if poison:
            return poison
        if input_queue.empty():
            time.sleep(step)
        else:
            return input_queue.get_nowait()


def _check_for_poison(poison_queue: Queue) -> Union[Poison, None]:
    """
    Check if poison has been added to the queue.
    """
    Logger.debug("Checking for poison")
    try:
        poison = poison_queue.get_nowait()
    except Empty:
        return
    # Put the poison back in the queue for another process to check on
    Logger.error("Poisoned")
    poison_queue.put(poison)
    return Poison(poison)


def _wait_on_output_to_be_available(
    poison_queue: Queue,
    output_queue: Queue,
    raise_exception: bool = False,
    item=None
) -> Union[Poison, None]:
    """

    :param poison_queue:
    :param output_queue:
    :param item:
    :return:
    """
    killed = _check_for_poison(poison_queue)
    # Wait until output queue is empty to limit rate and memory use
    tic = default_timer()
    while output_queue.full():
        # Keep on checking while we wait
        killed = _check_for_poison(poison_queue)
        if killed:
            break
        waited = default_timer() - tic
        if waited > 60:
            Logger.debug("Waiting for output_queue to not be full")
            tic = default_timer()
    if not killed and item:
        output_queue.put_nowait(item)
    elif killed and raise_exception:
        raise killed
    return killed


def _get_detection_stream(
    template_channel_ids: List[tuple],
    client,
    input_time_queue: Queue,
    retries: int,
    min_gap: float,
    buff: float,
    output_filename_queue: Queue,
    poison_queue: Queue,
    temp_stream_dir: str,
    full_stream_dir: str = None,
    pre_process: bool = False,
    parallel_process: bool = True,
    process_cores: int = None,
    daylong: bool = False,
    overlap: Union[str, float] = "calculate",
    ignore_length: bool = False,
    ignore_bad_data: bool = False,
    filt_order: int = None,
    highcut: float = None,
    lowcut: float = None,
    samp_rate: float = None,
    process_length: float = None,
):
    """
    Get a stream to be used for detection from a client for a time period.

    This function is designed to be run continuously within a Process and will
    only stop when the next item in the input_time_queue is None.

    This function uses .get_waveforms_bulk to get a Stream from a Client.
    The specific time period to get data for is read from the input_time_queue.
    Once the data have been loaded into memory from the Client, this function
    then processes that Stream according to the processing parameters passed
    as arguments to this function. Finally, this function writes the processed
    Stream to disk in a temporary stream directory and puts the filename
    for that stream in the output_filename_queue.

    Optionally, an unprocessed version of the stream can be written to the
    full_stream_dir directory to later provide an unprocessed copy of the
    raw data. This can be helpful if data downloading is slow and the stream
    is required for subsequent processing.

    :param template_channel_ids:
        Iterable of (network, station, location, channel) tuples to get data
        for. Wildcards may be used if accepted by the client.
    :param client:
        Client-like object with at least a .get_waveforms_bulk method.
    :param input_time_queue:
        Queue of (starttime, endtime) tuples of UTCDateTimes to get data
        between.
    :param retries: See core.match_filter.tribe.client_detect
    :param min_gap: See core.match_filter.tribe.client_detect
    :param buff:
        Length to pad downloaded data by - some clients do not provide all
        data requested.
    :param output_filename_queue:
        Queue to put filenames of written streams into
    :param poison_queue:
        Queue to check for poison, or put poison into if something goes awry
    :param temp_stream_dir:
        Directory to write processed streams to.
    :param full_stream_dir:
        Directory to save unprocessed streams to. If None, will not be used.
    :param pre_process: Whether to run pre-processing or not.
    :param parallel_process:
        Whether to process data in parallel (uses multi-threading)
    :param process_cores:
        Maximum number of cores to use for parallel processing
    :param daylong: See utils.pre_processing.multi_process
    :param overlap: See core.match_filter.tribe.detect
    :param ignore_length: See utils.pre_processing.multi_process
    :param ignore_bad_data: See utils.pre_processing.multi_process
    :param filt_order: See utils.pre_processing.multi_process
    :param highcut: See utils.pre_processing.multi_process
    :param lowcut: See utils.pre_processing.multi_process
    :param samp_rate: See utils.pre_processing.multi_process
    :param process_length: See utils.pre_processing.multi_process
    """
    while True:
        killed = _wait_on_output_to_be_available(
            poison_queue=poison_queue, output_queue=output_filename_queue,
            item=False)
        if killed:
            Logger.error("Killed")
            break
        try:
            next_times = _get_and_check(input_time_queue, poison_queue)
            if next_times is None:
                break
            if isinstance(next_times, Poison):
                Logger.error("Killed")
                break
            starttime, endtime = next_times

            st = _download_st(
                starttime=starttime, endtime=endtime, buff=buff,
                min_gap=min_gap, template_channel_ids=template_channel_ids,
                client=client, retries=retries)
            if len(st) == 0:
                Logger.warning(f"No suitable data between {starttime} "
                               f"and {endtime}, skipping")
                continue
            Logger.info(f"Downloaded stream of {len(st)} traces:")
            for tr in st:
                Logger.info(tr)
            # Try to reduce memory consumption by getting rid of st if we can
            if full_stream_dir:
                for tr in st:
                    tr.split().write(os.path.join(
                        full_stream_dir,
                        f"full_trace_{tr.id}_"
                        f"{tr.stats.starttime.strftime('%y-%m-%dT%H-%M-%S')}"
                        f".ms"), format="MSEED")
            if not pre_process:
                st_chunks = [st]
            else:
                template_ids = set(['.'.join(sid)
                                    for sid in template_channel_ids])
                # Group_process copies the stream.
                st_chunks = _pre_process(
                    st=st, template_ids=template_ids, pre_processed=False,
                    filt_order=filt_order, highcut=highcut,
                    lowcut=lowcut, samp_rate=samp_rate,
                    process_length=process_length,
                    parallel=parallel_process, cores=process_cores,
                    daylong=daylong, ignore_length=ignore_length,
                    overlap=overlap, ignore_bad_data=ignore_bad_data)
                # We don't need to hold on to st!
                del st
            for chunk in st_chunks:
                Logger.info(f"After processing stream has {len(chunk)} traces:")
                for tr in chunk:
                    Logger.info(tr)
                if not os.path.isdir(temp_stream_dir):
                    os.makedirs(temp_stream_dir)
                chunk_file = os.path.join(
                    temp_stream_dir,
                    f"chunk_{len(chunk)}_"
                    f"{chunk[0].stats.starttime.strftime('%Y-%m-%dT%H-%M-%S')}"
                    f"_{os.getpid()}.pkl")
                # Add PID to cope with multiple instances operating at once
                _pickle_stream(chunk, chunk_file)
                # Wait for output queue to be ready
                _wait_on_output_to_be_available(
                    poison_queue=poison_queue,
                    output_queue=output_filename_queue,
                    item=chunk_file, raise_exception=True)
                del chunk
        except Exception as e:
            Logger.error(f"Caught exception {e} in downloader")
            poison_queue.put(Poison(e))
            traceback.print_tb(e.__traceback__)
            break
    # Wait for output queue to be ready
    killed = _wait_on_output_to_be_available(
        poison_queue=poison_queue,
        output_queue=output_filename_queue,
        raise_exception=False)
    if killed:
        poison_queue.put_nowait(killed)
    else:
        output_filename_queue.put(None)
    return


def _pre_processor(
    input_stream_queue: Queue,
    temp_stream_dir: str,
    template_ids: set,
    pre_processed: bool,
    filt_order: int,
    highcut: float,
    lowcut: float,
    samp_rate: float,
    process_length: float,
    parallel: bool,
    cores: int,
    daylong: bool,
    ignore_length: bool,
    overlap: float,
    ignore_bad_data: bool,
    output_filename_queue: Queue,
    poison_queue: Queue,
):
    """
    Consume a queue of input streams and process those streams.

    This function is designed to be run continuously within a Process and will
    only stop when the next item in the input_stream_queue is None.

    This function consumes streams from the input_stream_queue and processes
    them using utils.pre_processing functions. Processed streams are written
    out to the temp_stream_dir and the filenames are produced in the
    output_filename_queue.

    :param input_stream_queue:
        Input Queue to consume streams from.
    :param temp_stream_dir: Directory to write processed streams to.
    :param template_ids:
        Iterable of seed ids in the template set. Only channels matching these
        seed ids will be retained.
    :param pre_processed: See core.match_filter.tribe.detect
    :param filt_order: See utils.pre_processing.multi_process
    :param highcut: See utils.pre_processing.multi_process
    :param lowcut: See utils.pre_processing.multi_process
    :param samp_rate: See utils.pre_processing.multi_process
    :param process_length: See utils.pre_processing.multi_process
    :param parallel: See utils.pre_processing.multi_process
    :param cores: See utils.pre_processing.multi_process
    :param daylong: See utils.pre_processing.multi_process
    :param ignore_length: See utils.pre_processing.multi_process
    :param overlap: See core.match_filter.tribe.detect
    :param ignore_bad_data: See utils.pre_processing.multi_process
    :param output_filename_queue:
        Queue to put filenames of written streams into
    :param poison_queue:
         Queue to check for poison, or put poison into if something goes awry
    """
    while True:
        killed = _check_for_poison(poison_queue)
        if killed:
            break
        Logger.debug("Getting stream from queue")
        st = _get_and_check(input_stream_queue, poison_queue)
        if st is None:
            Logger.info("Ran out of streams, stopping processing")
            break
        elif isinstance(st, Poison):
            Logger.error("Killed")
            break
        if len(st) == 0:
            break
        Logger.info(f"Processing stream:\n{st}")

        # Process stream
        try:
            st_chunks = _pre_process(
                st, template_ids, pre_processed, filt_order, highcut, lowcut,
                samp_rate, process_length, parallel, cores, daylong,
                ignore_length, ignore_bad_data, overlap)
            for chunk in st_chunks:
                if not os.path.isdir(temp_stream_dir):
                    os.makedirs(temp_stream_dir)
                chunk_file = os.path.join(
                    temp_stream_dir,
                    f"chunk_{len(chunk)}_"
                    f"{chunk[0].stats.starttime.strftime('%Y-%m-%dT%H-%M-%S')}"
                    f"_{os.getpid()}.pkl")
                # Add PID to cope with multiple instances operating at once
                _pickle_stream(chunk, chunk_file)
                # Wait for output queue to be ready
                _wait_on_output_to_be_available(
                    poison_queue=poison_queue,
                    output_queue=output_filename_queue,
                    item=chunk_file, raise_exception=True)
                del chunk
        except Exception as e:
            Logger.error(
                f"Caught exception in processor:\n {e}")
            poison_queue.put_nowait(Poison(e))
            traceback.print_tb(e.__traceback__)
    # Wait for output queue to be ready
    killed = _wait_on_output_to_be_available(
        poison_queue=poison_queue,
        output_queue=output_filename_queue,
        raise_exception=False)
    if killed:
        poison_queue.put_nowait(killed)
    else:
        output_filename_queue.put_nowait(None)
    return


def _prepper(
    input_stream_filename_queue: Queue,
    templates: Union[List, dict],
    group_size: int,
    groups: Iterable[Iterable[str]],
    output_queue: Queue,
    poison_queue: Queue,
    xcorr_func: str = None,
):
    """
    Prepare templates and stream for correlation.

    This function is designed to be run continuously within a Process and will
    only stop when the next item in the input_stream_queue is None.

    This function prepares (reshapes into numpy arrays) templates and streams
    and ensures that the data are suitable for the cross-correlation function
    specified.

    :param input_stream_filename_queue:
        Input Queue to consume stream_filenames from.
    :param templates:
        Either (a) a list of Template objects, or (b) a dictionary of pickled
        template filenames, keyed by template name.
    :param group_size:
        See core.match_filter.tribe.detect
    :param groups:
        Iterable of groups, where each group is an iterable of the template
        names in that group.
    :param output_queue:
        Queue to produce inputs for correlation to.
    :param poison_queue:
        Queue to check for poison, or put poison into if something goes awry
    :param xcorr_func:
        Name of correlation function backend to be used.
    """
    if isinstance(templates, dict):
        # We have been passed a db of template files on disk
        Logger.info("Deserializing templates from disk")
        try:
            templates = _read_template_db(templates)
        except Exception as e:
            Logger.error(f"Could not read from db due to {e}")
            poison_queue.put_nowait(Poison(e))
            return

    while True:
        killed = _check_for_poison(poison_queue)
        if killed:
            Logger.info("Killed in prepper")
            break
        Logger.info("Getting stream from queue")
        st_file = _get_and_check(input_stream_filename_queue, poison_queue)
        if st_file is None:
            Logger.info("Got None for stream, prepper complete")
            break
        elif isinstance(st_file, Poison):
            Logger.error("Killed")
            break
        if isinstance(st_file, Stream):
            Logger.info("Stream provided")
            st = st_file
            # Write temporary cache of file
            st_file = tempfile.NamedTemporaryFile().name
            Logger.info(f"Writing temporary stream file to {st_file}")
            try:
                _pickle_stream(st, st_file)
            except Exception as e:
                Logger.error(
                    f"Could not write temporary file {st_file} due to {e}")
                poison_queue.put_nowait(Poison(e))
                break
        Logger.info(f"Reading stream from {st_file}")
        try:
            st = _unpickle_stream(st_file)
        except Exception as e:
            Logger.error(f"Error reading {st_file}: {e}")
            poison_queue.put_nowait(Poison(e))
            break
        st_sids = {tr.id for tr in st}
        if len(st_sids) < len(st):
            _sids = [tr.id for tr in st]
            _duplicate_sids = {
                sid for sid in st_sids if _sids.count(sid) > 1}
            poison_queue.put_nowait(Poison(NotImplementedError(
                f"Multiple channels in continuous data for "
                f"{', '.join(_duplicate_sids)}")))
            break
        # Do the grouping for this stream
        Logger.info(f"Grouping {len(templates)} templates into groups "
                    f"of {group_size} templates")
        try:
            template_groups = _group(sids=st_sids, templates=templates,
                                     group_size=group_size, groups=groups)
        except Exception as e:
            Logger.error(e)
            poison_queue.put_nowait(Poison(e))
            break
        Logger.info(f"Grouped into {len(template_groups)} groups")
        for i, template_group in enumerate(template_groups):
            killed = _check_for_poison(poison_queue)
            if killed:
                break
            try:
                template_streams = [
                    _quick_copy_stream(t.st) for t in template_group]
                template_names = [t.name for t in template_group]

                # template_names, templates = zip(*template_group)
                Logger.info(
                    f"Prepping {len(template_streams)} "
                    f"templates for correlation")
                # We can just load in a fresh copy of the stream!
                _st, template_streams, template_names = \
                    _prep_data_for_correlation(
                        stream=_unpickle_stream(st_file).merge(),
                        templates=template_streams,
                        template_names=template_names)
                if len(_st) == 0:
                    Logger.error(
                        f"No traces returned from correlation prep: {_st}")
                    continue
                starttime = _st[0].stats.starttime

                if xcorr_func in (None, "fmf", "fftw"):
                    array_dict_tuple = _get_array_dicts(
                        template_streams, _st, stack=True)
                    stream_dict, template_dict, pad_dict, \
                        seed_ids = array_dict_tuple
                    if xcorr_func == "fmf":
                        Logger.info("Prepping data for FMF")
                        # Work out used channels here
                        tr_chans = np.array(
                            [~np.isnan(template_dict[seed_id]).any(axis=1)
                             for seed_id in seed_ids])
                        no_chans = np.sum(np.array(tr_chans).astype(int),
                                          axis=0)
                        chans = [[] for _i in range(len(templates))]
                        for seed_id, tr_chan in zip(seed_ids, tr_chans):
                            for chan, state in zip(chans, tr_chan):
                                if state:
                                    chan.append((seed_id.split('.')[1],
                                                 seed_id.split('.')[-1].split(
                                                     '_')[0]))
                        # Reshape
                        t_arr, d_arr, weights, pads = _fmf_reshape(
                            template_dict=template_dict,
                            stream_dict=stream_dict,
                            pad_dict=pad_dict, seed_ids=seed_ids)
                        # Stabilise
                        t_arr, d_arr, multipliers = _fmf_stabilisation(
                            template_arr=t_arr, data_arr=d_arr)
                        # Wait for output queue to be ready
                        _wait_on_output_to_be_available(
                            poison_queue=poison_queue,
                            output_queue=output_queue,
                            item=(starttime, i, d_arr, template_names, t_arr,
                                  weights, pads, chans, no_chans),
                            raise_exception=True)
                    else:
                        Logger.info("Prepping data for FFTW")
                        # Wait for output queue to be ready
                        killed = _wait_on_output_to_be_available(
                            poison_queue=poison_queue,
                            output_queue=output_queue,
                            item=(starttime, i, stream_dict, template_names,
                                  template_dict, pad_dict, seed_ids),
                            raise_exception=True)
                else:
                    Logger.info("Prepping data for standard correlation")
                    # Wait for output queue to be ready
                    killed = _wait_on_output_to_be_available(
                        poison_queue=poison_queue, output_queue=output_queue,
                        item=(starttime, i, _st, template_names,
                              template_streams),
                        raise_exception=True)
            except Exception as e:
                Logger.error(f"Caught exception in Prepper: {e}")
                traceback.print_tb(e.__traceback__)
                poison_queue.put_nowait(Poison(e))
            i += 1
        Logger.info(f"Removing temporary {st_file}")
        os.remove(st_file)
    # Wait for output queue to be ready
    killed = _wait_on_output_to_be_available(
        poison_queue=poison_queue, output_queue=output_queue,
        raise_exception=False)
    if killed:
        poison_queue.put_nowait(killed)
    else:
        output_queue.put_nowait(None)
    return


def _make_detections(
    input_queue: Queue,
    delta: float,
    templates: Union[List, dict],
    threshold: float,
    threshold_type: str,
    save_progress: bool,
    output_queue: Queue,
    poison_queue: Queue,
):
    """
    Construct Detection objects from sparse detection information.

    This function is designed to be run continuously within a Process and will
    only stop when the next item in the input_queue is None.

    :param input_queue:
        Queue of (starttime, peaks, thresholds, no_channels, channels,
        template_names). Detections are made within `peaks`.
    :param delta:
        Sample rate of peaks to detect within in Hz
    :param templates:
        Template objects included in input_queue
    :param threshold:
        Overall threshold
    :param threshold_type:
        Overall threshold type
    :param save_progress:
        Whether to save progress or not: If true, individual Party files will
        be written each time this is run.
    :param output_queue:
        Queue of output Party filenames.
    :param poison_queue:
        Queue to check for poison, or put poison into if something goes awry
    """
    chunk_id = 0
    while True:
        killed = _check_for_poison(poison_queue)
        if killed:
            break
        try:
            next_item = _get_and_check(input_queue, poison_queue)
            if next_item is None:
                Logger.info("_make_detections got None, stopping")
                break
            elif isinstance(next_item, Poison):
                Logger.error("Killed")
                break
            starttime, all_peaks, thresholds, no_chans, \
                chans, template_names = next_item
            detections = _detect(
                template_names=template_names, all_peaks=all_peaks,
                starttime=starttime, delta=delta, no_chans=no_chans,
                chans=chans, thresholds=thresholds)
            Logger.info(f"Built {len(detections)}")
            chunk_file = _make_party(
                detections=detections, threshold=threshold,
                threshold_type=threshold_type, templates=templates,
                chunk_start=starttime, chunk_id=chunk_id,
                save_progress=save_progress)
            chunk_id += 1
            output_queue.put_nowait(chunk_file)
        except Exception as e:
            Logger.error(
                f"Caught exception in detector:\n {e}")
            traceback.print_tb(e.__traceback__)
            poison_queue.put_nowait(Poison(e))
    output_queue.put_nowait(None)
    return


if __name__ == "__main__":
    import doctest

    doctest.testmod()
