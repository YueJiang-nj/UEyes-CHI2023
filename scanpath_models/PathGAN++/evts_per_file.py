#!/usr/bin/env python3
# coding: utf-8

'''
Creates a consolidated saliency dataset from the files created by the SMI RED250 eye-tracker.

Usage: evts_per_file.py user_events_dir fixation_events_dir

External dependencies, to be installed e.g. via pip:
- none

Authors:
- Luis A. Leiva <name.surname@uni.lu>
- Paul Houssel <name.surname.001@uni.lu>
'''

import sys
import os
import re

re_user = re.compile(r'U[0-9]+')

def process_user_events_line(line):
    # Each row is: `UserEvent,1,1,4419737762,# Message: hw_2.jpg`
    _, trial, num, timestamp, msg = line.strip().split(',')
    ui_filename = msg.replace('# Message: ', '')
    return int(timestamp), ui_filename


def process_fixation_events_line(line):
    # Each row is: `Fixation L,1,1,4419745615,4419970873,225258,946.43,437.34,19,16,-1,12.40,12.40`
    _, trial, num, ini_t, end_t, duration, x, y, d_x, d_y, plane, pupil_h, pupil_v = line.strip().split(',')
    return int(ini_t), int(end_t), int(duration), float(x), float(y)


def process_user_events(filepath):
    tuples = []
    with open(filepath) as f:
        # FIXME: `splitlines()` put all lines into memory, so it's very inefficient.
        # However we need to peek the next line to get the ending timestamp.
        lines = f.read().splitlines()
        for n, line in enumerate(lines):
            if not line.startswith('UserEvent'):
                continue
            ini_timestamp, ui_filename = process_user_events_line(line)
            if n < len(lines) - 1:
                end_timestamp, _ = process_user_events_line(lines[n + 1])
            else:
                # FIXME: How can we estimate the duration of the last event?
                # Actually it doesn't matter, but we can add a large number to be on the safe side.
                end_timestamp = ini_timestamp * 10
            tuples.append((ini_timestamp, end_timestamp, ui_filename))
    return tuples


def process_fixation_events(filepath):
    entries = []
    with open(filepath) as f:
        for line in f:
            if not line.startswith('Fixation L'): # and not line.startswith('Fixation R'):
                continue
            ini_t, end_t, duration, x, y = process_fixation_events_line(line)
            entries.append({
                'ini_t': ini_t,
                'end_t': end_t,
                'duration': duration,
                'x': x,
                'y': y,
            })
    return entries


if __name__ == '__main__':
    user_events_dir = sys.argv[1]
    fixation_events_dir = sys.argv[2]
    # Define if the duration feature shall be taken into account or not.
    reduced = sys.argv[3] if len(sys.argv) > 3 else "true"

    # XXX: Sorting the full output via Bash will put the header elsewhere. Use this:
    # ~$ awk 'NR == 1; NR > 1{ print $0 | "sort -n" }' dataset.csv > dataset_sorted.csv
    if reduced == "true":
        print('ui_file,username,x,y,timestamp')
    else:
        print('ui_file,username,x,y,timestamp,duration')

    for path, directories, files in os.walk(user_events_dir):
        for f in files:
            if not f.endswith('.txt'):
                continue
            evt_file = os.path.join(path, os.path.basename(f))
            eye_file = os.path.join(fixation_events_dir, os.path.basename(f))
            username = re_user.findall(evt_file)[0]
            user_events = process_user_events(evt_file)
            fixation_events = process_fixation_events(eye_file)
            for (ini_t, end_t, ui_file) in user_events:
                for entry in fixation_events:
                    if entry['ini_t'] > ini_t and entry['end_t'] < end_t:
                        if reduced == "true":
                            print("{},{},{},{},{}".format(ui_file, username, entry['x'], entry['y'], entry['ini_t']))
                        else:
                            print("{},{},{},{},{},{}".format(ui_file, username, entry['x'], entry['y'], entry['ini_t'], entry['duration']))
