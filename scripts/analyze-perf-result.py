#!/usr/bin/env python

from __future__ import print_function
import sys
import string
import numpy


def main():
    if len(sys.argv) < 2:
        print("Usage: ./analyze-perf-result.py perf_file1 perf_file_2 ...", file=sys.stderr)
        sys.exit(1)

    data = dict()

    for input_filename in sys.argv[1:]:
        print("Processing %s ..." % (input_filename, ), file=sys.stderr)

        with open(input_filename) as input_file:
            for line in input_file:
                if string.find(line, "(ms)") == -1:
                    continue
                elements = string.split(line)
                #assert elements[2] == "(ms)"
                if not elements[2] == "(ms)":
                    continue
                component = elements[0]
                #print(line, elements)
                exec_time = float(elements[1])
                if not component in data:
                    data[component] = list()

                data[component].append(exec_time)

    print("#component, mean exec time, stddev exec time, data points")
    for component, exec_time_list in data.items():
        numd = len(exec_time_list)
        mean = numpy.mean(exec_time_list)
        stdd = numpy.std(exec_time_list)
        print("%s, %s, %s, %s " % (component, mean, stdd, numd))

if __name__ == "__main__":
    main()
