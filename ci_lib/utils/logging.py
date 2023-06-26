'''
Contains logging configuration functions for snakemake workflow scripts
and a class usefull to redirect std streams to loggers
'''

import sys
import logging

LOGLEVELS = {
            "DEBUG":logging.DEBUG,
            "INFO":logging.INFO,
            "WARNING":logging.WARNING,
            "ERROR":logging.ERROR,
            "CRITICAL":logging.CRITICAL, }

def start_log(snakemake):
    '''
    Configure logging to log into the snakemake specified logfile
    '''
    # logging <3.9 does not support encoding
    if sys.version_info[0] == 3 and sys.version_info[1] < 9 :
        logging.basicConfig(filename=str(snakemake.log), style="{",
                format="{asctime} {name}: {levelname}: {message}", datefmt="%b %d %H:%M:%S",
                level=LOGLEVELS[snakemake.config["loglevel"]])
    else:
        logging.basicConfig(filename=str(snakemake.log), encoding='utf-8', style="{",
                format="{asctime} {name}: {levelname}: {message}", datefmt="%b %d %H:%M:%S",
                level=LOGLEVELS[snakemake.config["loglevel"]])
    logger = logging.getLogger(f"{snakemake.rule}")
    logger.info("Start of rule")
    logger.info("Loglevel: %s", logger.getEffectiveLevel())
    return logger

class StreamToLogger:
    '''
    Fake file-like stream object that redirects writes to a logger instance.
    '''
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        '''Write the buffer through to the logger'''
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        '''Noop'''
