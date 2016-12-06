#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com>
# modifications by: Luis Castro and Jens Röwekamp

import os
import sys
import time

class Logger(object):

    def __init__(self, log_dir, game_name, debug=False, verbosity=0):
        self.log_dir = log_dir
        self._game_name = game_name
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self._debug = debug
        self.DATE_FORMAT = "%Y-%m-%d"
        self.DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
        self._log_date = self._curdate()
        self._logfile = "%s/%s_%s.log" % (self.log_dir, self._game_name, int(time.time()))
        self._expdatafile = "%s/%s_%s.csv" % (self.log_dir, self._game_name, int(time.time()))
        self._logger = open(self._logfile, 'a+')
        self._expdata = open(self._expdatafile, 'a+')
        self._verbosity = verbosity

    def _curdate(self):
        return time.strftime(self.DATE_FORMAT, time.localtime())

    def _curdatetime(self):
        return time.strftime(self.DATETIME_FORMAT, time.localtime())

    def _switch_log(self):
        if self._log_date != self._curdate():  # create new logfile
            # close old logfile
            self._logger.close()
            # make new log file
            self._log_date = self._curdate()
            self._logfile = "%s/%s.log" % (self.log_dir, self._log_date)
            self._logger = open(self._logfile, 'a+')

    def _writer(self, msg):
        #self._switch_log()
        # maybe locker is needed here
        self._logger.write("%s\n" % msg)

    def debug(self, msg):
        if self._debug:
            msg = "%s [DEBUG] %s" % (self._curdatetime(), msg)
            self._writer(msg)

    def info(self, msg):
        if self._verbosity < 30:
            msg = "%s [INFO] %s" % (self._curdatetime(), msg)
            if (self._verbosity > 0):
                print msg
            self._writer(msg)

    def warn(self, msg):
        msg = "%s [WARN] %s" % (self._curdatetime(), msg)
        # print msg
        self._writer(msg)

    def error(self, msg, to_exit=False):
        msg = "%s [ERROR] %s" % (self._curdatetime(), msg)
        print msg
        self._writer(msg)
        if to_exit:
            sys.exit(-1)

    def exp(self, data_line):
        if self._verbosity < 30:
            msg = ""
            if len(data_line) > 1:
                for sample in range(0,(len(data_line) - 1)):
                    msg += "%s," % (str(data_line[sample]))
                msg += "%s" % (str(data_line[-1]))
            elif len(data_line) == 1:
                msg = "%s" % (str(data_line[0]))
            self._expdata.write("%s\n" % msg)

    def exp2(self, data_line):
        msg = ""
        if len(data_line) > 1:
            for sample in range(0,(len(data_line) - 1)):
                msg += "%s," % (str(data_line[sample]))
            msg += "%s" % (str(data_line[-1]))
        elif len(data_line) == 1:
            msg = "%s" % (str(data_line[0]))
        self._expdata.write("%s\n" % msg)

    def close_log(self):
        self._logger.close()

#logger = Logger(log_dir="./log")
