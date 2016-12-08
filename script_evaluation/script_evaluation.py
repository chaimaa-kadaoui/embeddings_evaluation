#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
One shot script for evaluating word embeddings.
This script was created to work for python3.4. No guaranty is given with regards to it working for 
an older version, especially python2.

It is assumed that the embedding to be evaluated exists somewhere locally under the form of two files:
- vocabulary.txt (one word per line)
- matrix.npy (for dense matrix as numpy.array, for instance created using numpy.save()) or 
matrix.npz (for scipy.sparse.csr_matrix, saved using the homemade function 'save_sparse_csr')

To evaluate this embedding using the script, feed this script the path to the folder where those 
two files are located together.

Use 'python3

@author: François Noyez
'''

import logging
import numpy as np
import scipy.stats
import os.path
import numpy
from scipy import sparse
from collections import defaultdict
import shutil
import itertools
from copy import deepcopy as copy_deepcopy
numpy.seterr(all='warn') # So that numpy 'warnings' are true 'warnings', and not just print to stdout or stderr
from scipy import sparse
from heapq import nlargest
import multiprocessing.queues
from multiprocessing import Process, Queue, cpu_count
import re
import tempfile
import gzip
import traceback
import sklearn.preprocessing
import time
import pickle
from logging import DEBUG, INFO, ERROR, WARN, WARNING, CRITICAL, FATAL
LOGGING_LEVELS = {"debug": DEBUG, "info": INFO, "error": ERROR, "warn": WARN
                  , "warning":WARNING, "error": ERROR, "critical":CRITICAL
                  , "fatal": FATAL}
POSSIBLE_LOGGING_LEVELS_KEYS = str(sorted(list(LOGGING_LEVELS.keys())))

ENCODING = "utf-8"
DEFAULT_LABEL = "No label provided."

ARCHIVE_SUFFIX = ".zip"
VOCABULARY_FILE_NAME = "vocabulary.txt"
PICKLE_MATRIX_FILE_NAME = "matrix.pkl"
NUMPY_MATRIX_FILE_NAME = "matrix.npy"
SPARSE_MATRIX_FILE_NAME = "sparse_matrix.npz"
CREATION_PARAMETERS_MAP_TXT_FILE_NAME = "creation_parameters.txt"
CREATION_PARAMETERS_MAP_PICKLE_FILE_NAME = "creation_parameters.pickle"

EVALUATION_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation")


def format_logging_level(logging_level):
    """Convert strings representing logging levels of the 'logging' python module to their proper
    integer value.

    :param logging_level: string (such as 'error', 'debug', 'info'...) or integer
    :return:
    """
    if isinstance(logging_level, str):
        logging_lvl = logging_level.lower()
        if logging_lvl not in LOGGING_LEVELS:
            error_message = "'{:s}' is not a correct logging level name (possible names are '{:s}')"
            error_message = error_message.format(logging_level, POSSIBLE_LOGGING_LEVELS_KEYS)
            raise ValueError(error_message)
        return LOGGING_LEVELS[logging_lvl]

    if not isinstance(logging_level, int):
        error_message = "Incorrect logging level type ('{}'), must be int or a correct string "\
                        "(possible strings are '{:s}')."
        error_message = error_message.format(type(logging_level), POSSIBLE_LOGGING_LEVELS_KEYS)
        raise ValueError(error_message)

    return logging_level

EPSILON_COSMUL = 0.001
POSSIBLE_ANALOGY_TYPE = ["cosmul", "additive"]
ANALOGY_OWN_ENGLISH_DATASETS = ["google", "msr"]

REPORT_FILENAME = "embedding_evaluation_report.txt"
POSSIBLE_LANGUAGE_LIST = ["en", "fr", "de"]
NO_FILE_WARNING_MESSAGE_TEMPLATE = "No evaluation data files in folder '{:s}'."

REPORT_TITLE = "\n".join(["#####################", "# Evaluation report #", "#####################"])
ANALOGY_SECTION_TITLE = "\n".join(["###########", "# Analogy #", "###########"])
SIMILARITY_REPORT_TITLE = "\n".join(["##############", "# Similarity #", "##############"])

TEMPORARY_EMBEDDING_NAME_PREFIX = "emb"


logger = logging.getLogger(__name__)


###############################
# Utility functions & classes #
###############################
class Vocabulary:
    
    def __init__(self, source, **kwargs):
        self._logger = logging.getLogger("{}.{}".format(__name__, self.__class__.__name__))
        self.index_word = []
        self.word_index = {}
        self._factory(source, **kwargs)

    def __len__(self):
        return len(self.index_word)

    def index(self, word):
        """Returns the index associated to the word"""
        return self.word_index[word]

    def words(self):
        """Returns the words of the vocabulary as a list"""
        return list(self.word_index.keys())

    def save(self, file_path):
        """Save the vocabulary in a file.

            :type file_path: string
            :param file_path: Local path to file where vocabulary should be written
            :return: None

            .. seealso:: load()
            ..warning:: If the file already exists before asking the vocabulary
            to save_to_file, it will be overwritten.
        """
        with open(file_path, "w", encoding=ENCODING) as f:
            f.write("\n".join(self.index_word))

    def _factory(self, words, **kwargs):
        if isinstance(words, self.__class__):
            self._from_list(words.index_word)
        elif isinstance(words, list):
            self._from_list(words)
        else:
            raise NotImplementedError

    def _from_list(self, words):
        self.word_index = {word: index for index, word in enumerate(words)}
        words = sorted(self.word_index.keys(), key=lambda w: self.word_index[w])
        self.index_word = words
        
    @staticmethod
    def load(file_path):
        """Load the vocabulary from its associated file.

            :type file_path: string
            :param file_path: Local path to file where vocabulary is located
            :return: a Vocabulary instance

            .. seealso:: save()
        """
        temp_words = []
        with open(file_path, "r", encoding=ENCODING) as f:
            for word in f:
                word = word.strip()
                if word != "":
                    temp_words.append(word)
        return Vocabulary(temp_words)


class Embedding:
    
    _logger = logging.getLogger("{}.{}".format(__name__, "Embedding"))
    
    def __init__(self, words_vocabulary, matrix, label=DEFAULT_LABEL):
        self._set_words_vocabulary_and_matrix(words_vocabulary, matrix)
        self.label = label
    
    # Instance attributes
    @property
    def vocabulary(self):
        return self._vocabulary
    @vocabulary.setter
    def vocabulary(self, a_vocabulary):
        error_message = "Error: you cannot directly set the vocabulary. If you want "\
        "to set it anew, you must use a 'load_from_matrix_and_...' method of "\
        "the embedding object."
        raise Exception(error_message)
    
    @property
    def matrix(self):
        return self._matrix
    @matrix.setter
    def matrix(self, a_matrix):
        error_message = "Error: you cannot directly set the matrix. If you want "\
        "to set it anew, you must use a 'load_from_matrix_and_...' method of "\
        "the embedding object."
        raise Exception(error_message)
    
    @property
    def l2_normalized_matrix(self):
        return self._l2_normalized_matrix
    @l2_normalized_matrix.setter
    def l2_normalized_matrix(self, a_matrix):
        error_message = "Error: you cannot set the normalized matrix. You must"+\
        " first set the embedding matrix, then call the 'l2_normalize' method of "+\
        "the Embedding object."
        raise Exception(error_message)
    
    @property
    def label(self):
        return self._label
    @label.setter
    def label(self, label):
        self._label = "{:s} ".format(label).strip()
    
    # Instance methods
    def _set_words_vocabulary_and_matrix(self, words_vocabulary, matrix):
        """
            Check input and consistency between them.
            If everything is correct, a copy of both inputs is made, and each 
            of the respective copy is assigned to be the corresponding 
            attribute value of the Embedding object.
        """
        # Check the vocabulary
        _words_vocabulary = Vocabulary(words_vocabulary)
        
        # Copy the input matrix
        _matrix = copy_deepcopy(matrix)
        
        # Assign values
        self._matrix = _matrix
        self._l2_normalized_matrix = None
        self._vocabulary = _words_vocabulary
    
    def is_l2_normalized(self):
        if self.l2_normalized_matrix is None:
            return False
        else:
            return True
    
    def l2_normalize(self):
        if self.matrix is None:
            error_message = "The embedding matrix is not defined, cannot l2_normalize it."
            raise AttributeError(error_message)
        else:
            self._l2_normalized_matrix = l2_normalize(self.matrix, True)
    
    def copy(self):
        return Embedding(self.vocabulary, self.matrix, label=self.label)
    
    def get_pruned_copy(self, max_vocab_size):
        word_to_keep_nb = int(max_vocab_size)
        if word_to_keep_nb < 1:
            error_message = "Input word_to_keep_nb is too small ({:d}), must be strictly higher than 0."
            error_message = error_message.format(word_to_keep_nb)
            raise ValueError(error_message)
        
        current_vocab_size = len(self.vocabulary)
        if (word_to_keep_nb >= current_vocab_size): # the task exists only if we want to keep less items than there is
            new_embedding = self.copy()
        
        else:
            # Get pruned words list
            kept_words_list = self.vocabulary.index_word[:word_to_keep_nb]
            # Create the corresponding map while keeping the old index values
            reduced_matrix = self.matrix[:word_to_keep_nb,:]
            # Create the corresponding embedding
            new_embedding = Embedding(kept_words_list, reduced_matrix, label=self.label)
        
        return new_embedding
    
    def save_in_folder(self, folder_path):
        logger = self._logger
        
        # Make the folder exist if it does not and / or check for write permission
        check_and_create_folder(folder_path, need_write=True, verbose=False)
        
        with InputOutputLoggerTimer(logger, "debug", 
                                    "saving the embedding to the folder '{:s}'".format(folder_path)):
            
            # Save the vocabulary
            vocabulary_file_path = os.path.join(folder_path, VOCABULARY_FILE_NAME)
            self.vocabulary.save(vocabulary_file_path)
            
            # Save the matrix
            if (isinstance(self.matrix, numpy.ndarray)):
                filepath = os.path.join(folder_path, NUMPY_MATRIX_FILE_NAME)
                numpy.save(filepath, self.matrix)
                    
            elif (sparse.issparse(self.matrix)):
                filepath = os.path.join(folder_path, SPARSE_MATRIX_FILE_NAME) 
                arrays.save_sparse_csr(filepath, self.matrix)
                
            else:
                filepath = os.path.join(folder_path, PICKLE_MATRIX_FILE_NAME) 
                with open(filepath, "wb") as f:
                    pickle.dump(self.matrix, f)
    
    def save_as_archive(self, archive_file_path, temp_folder_path=None):
        """
            Save the embedding as an uncompressed '"""+ARCHIVE_SUFFIX+"""' archive.
        """
        logger = self._logger
        
        if not archive_file_path.endswith(ARCHIVE_SUFFIX):
            new_file_path = "{}{}".format(archive_file_path, ARCHIVE_SUFFIX)
            error_message = "Wrong file path input ({:s}), must end with '{:s}', '{:s}' will be used instead."
            error_message = error_message.format(archive_file_path, ARCHIVE_SUFFIX, new_file_path)
            archive_file_path = new_file_path
            logger.warning(error_message)
        
        with InputOutputLoggerTimer(logger, "debug", 
                                    "saving the embedding to the archive '{:s}'".format(archive_file_path)):
            # Check that we can write in destination folder
            with zipfile.ZipFile(archive_file_path, 'w', zipfile.ZIP_STORED) as ziph:
                # Create temporary folder
                with TemporaryDirectory(temp_folder_path) as td:
                    temporary_folder_path = td.temporary_folder_path
                    # Store the embedding inside that folder
                    self.save_in_folder(temporary_folder_path)
                    # Write file in archive
                    for file_path in _list_files_root(temporary_folder_path):
                        ziph.write(file_path, arcname=os.path.basename(file_path))
        
        return archive_file_path
    
    # Class methods
    @classmethod
    def _load_embedding_matrix_from_folder(cls, folder_path):
        logger = cls._logger
        
        filepath_list = list_files_root(folder_path)
        filename_list = [os.path.basename(filepath) for filepath in filepath_list]
        if NUMPY_MATRIX_FILE_NAME in filename_list:
            filepath = os.path.join(folder_path, NUMPY_MATRIX_FILE_NAME)
            try:
                matrix = numpy.asarray(numpy.load(filepath))
            except OSError as e:
                # If we have the following error, then the file is probably the result of numpy.ndarray.dump, which is just a wrapper around a pickle file
                if str(e) == "Failed to interpret file '{:s}' as a pickle".format(filepath):
                    with open(filepath, "rb") as f:
                        matrix = numpy.asarray(pickle.load(f, encoding="latin1"))
                    warning_message = "Problem in 'numpy.load({:s})': in order to try to load its content, the file will now be interpreted as a latin1 encoded pickle file."\
                    "\n\t\t\t\t\tThis can happen if the file is the result of 'numpy.ndarray.dump()' in python2. "\
                    "To prevent such compatibility issues, prefer using 'numpy.save()' rather than 'numpy.ndarray.dump()' if possible."
                    warning_message = warning_message.format(filepath)
                    logger.warning(warning_message)
        
        elif SPARSE_MATRIX_FILE_NAME in filename_list:
            filepath = os.path.join(folder_path, SPARSE_MATRIX_FILE_NAME)
            matrix = load_sparse_csr(filepath)
            
        elif PICKLE_MATRIX_FILE_NAME in filename_list:
            filepath = os.path.join(folder_path, PICKLE_MATRIX_FILE_NAME)
            with open(filepath, "rb") as f:
                matrix = pickle.load(f)
            
        else:
            raise ValueError("No matrix file is present in the embedding's folder.")
        
        return matrix
    
    @classmethod
    def load_from_folder(cls, folder_path, label=DEFAULT_LABEL):
        logger = cls._logger
        
        if not os.path.isdir(folder_path):
            error_message = "No folder located at {:s}.".format(folder_path)
            raise ValueError(error_message)
        
        with InputOutputLoggerTimer(logger, "debug", 
                                    "loading the embedding located in the folder '{:s}'".format(folder_path)):
            # Load the vocabulary
            vocabulary_file_path = os.path.join(folder_path, VOCABULARY_FILE_NAME)
            words_vocabulary = Vocabulary.load(vocabulary_file_path)
            
            # Load the matrix
            matrix = cls._load_embedding_matrix_from_folder(folder_path)
            
            # Create the embedding
            embedding = Embedding(words_vocabulary, matrix, label=label)
            
        return embedding
    
    @classmethod
    def load_from_archive(cls, archive_file_path, label=DEFAULT_LABEL, temp_folder_path=None):
        """Load from a folder that has been transformed into a '"""+ARCHIVE_SUFFIX+"""' archive."""
        logger = cls._logger
        
        with InputOutputLoggerTimer(logger, "debug", 
                                    "loading the embedding located in the archive '{:s}'".format(archive_file_path)):
            # Check that we can write in destination folder
            with zipfile.ZipFile(archive_file_path, 'r', zipfile.ZIP_STORED) as ziph:
                # Create temporary folder
                with TemporaryDirectory(temp_folder_path) as td:
                    temporary_folder_path = td.temporary_folder_path
                    # Extract archive elements to temporary folder
                    ziph.extractall(path=temporary_folder_path)
                    # Load the embedding from this temporary folder
                    embedding = cls.load_from_folder(temporary_folder_path, label=label)
        
        return embedding
    
    @classmethod
    def load_from_text_file(cls, file_path, label=DEFAULT_LABEL):
        """Load an embedding from a text file, where there is one word and its corresponding list 
        of embedding values per line. Words and values must be separated by what python consider to 
        be a 'space' character, notably a blank ' ' or a tabulation '\t'.
        
        Gunzip-compressed files are also a valid input.
        :param file_path: path to the text file
        :param label: if you want to give a fancy name to your embedding...
        :returns: an Embedding instance
        """
        logger = cls._logger
        
        def _parse_line_for_dense(line, words_list, vector_list):
            word_string_numbers = line.split()
            words_list.append(word_string_numbers[0])
            vector_list.append(numpy.array(word_string_numbers[1:], dtype=numpy.float))
        
        # Set up differently in case a compressed file
        _get_open_function = lambda: open(file_path, 'r', encoding=ENCODING)
        appended = ""
        if (file_path.endswith(".gz")):
            _get_open_function = lambda: gzip.open(file_path, 'rt', encoding=ENCODING)
            appended = " compressed"
        
        with InputOutputLoggerTimer(logger, "debug", 
                                    "loading the embedding located in the {:s} text file '{:s}'".format(file_path, appended)):
            with  _get_open_function() as f:
                # Assume the content is the representation of a  numpy.array
                words_list = []
                vector_list = []
                
                # Parse the file
                for line in f:
                    _parse_line_for_dense(line, words_list, vector_list)
                
                # Create the matrix
                matrix = numpy.array(vector_list, dtype=numpy.float)
                    
        return Embedding(words_list, matrix, label=label)
                        



class InputOutputLoggerTimer(object):
    
    def __init__(self, logger=None, logging_level="debug", logging_message=None,
                 report_message=None, report_array=None):
        # Preprocess the logging_lvl value
        if logging_level:
            logging_lvl = format_logging_level(logging_level)

        # Prepare the logging methods
        if logger and logging_message:
            begin_message_function = \
             lambda: logger.log(logging_lvl, "Starting {:s}...".format(logging_message))
            end_message_function = \
             lambda: logger.log(logging_lvl, "Finished {:s}...".format(logging_message))
        else:
            begin_message_function = lambda: None
            end_message_function = begin_message_function
        self._begin_message_function = begin_message_function
        self._end_message_function = end_message_function

        # Prepare the duration logging and reporting if necessary
        def _reporting_function(sec_nb):
            if report_message:
                message = report_message.format(format_seconds_nb(sec_nb))
                if logger and logging_lvl is not None:
                    logger.log(logging_lvl, message)
                if report_array is not None:
                    report_array.append(message)
        self._reporting_function = _reporting_function

        # Initializing attributes values
        self.start = None
        self.end = None
        self.secs = None

    def __enter__(self):
        self._begin_message_function()
        self.start = time.time()
        return self

    def __exit__(self, *args):
        # Do not display anything if an exception occurred, otherwise it produces a strange result when reading the logs
        if not (args and ((isinstance(args[0], SystemExit) and args[1].args[0] != 0) or isinstance(args[1], Exception))):
            self.end = time.time()
            self._end_message_function()
            self.secs = self.end - self.start
            self._reporting_function(self.secs)
    

def format_seconds_nb(seconds_nb, decimals=-1):
    """Create a formatted string to represent a duration using the conventional time base system.

    For an input duration expressed in seconds, outputs a string representing this duration
    expressed in natural language using the time base system (from centuries to microseconds,
    anyway).
    :param seconds_nb: the number of second (can be int or float)
    :param decimals: number of decimals to keep before formatting, if 'seconds_nb' is a float.
    -1 for keeping all of them.
    :type decimals: int (default = -1)
    :return: the string representing the natural language version of the input duration

    :Example:

    >>> format_seconds_nb(3665.3987123, decimals=3)
    '1h 1m 5s 398ms'
    """
    # Check parameters
    decimals = int(decimals)

    if seconds_nb is None:
        return  ""

    if np.isnan(seconds_nb):
        return  "NaNs"

    the_sign = np.sign(seconds_nb)
    used_duration = the_sign * seconds_nb  # Making sure we work on a non-negative number
    if decimals > -1:
        used_duration = np.around(used_duration, decimals)

    if used_duration == 0:
        return "0s"

    # Prepare the output
    str_array = []

    # Part of duration that is over a second
    date_keys_list = [("C", 100 * 365.4 * 24 * 60 * 60), ("Y", 365.25 * 24 * 60 * 60), 
                      ("M", 30.5 * 24 * 60 * 60), ("D", 24 * 60 * 60), ("h", 60 * 60), 
                      ("m", 60), ("s", 1)]
    remaining_secs_nb = int(used_duration)
    for symb, nb_secs in date_keys_list:
        if remaining_secs_nb == 0:  # If the duration to format has become null, we finish the step early
            break
        nb_symb, remaining_secs_nb = divmod(remaining_secs_nb, nb_secs)
        if nb_symb > 0:
            to_add = "{:s}{:s}".format(str(int(the_sign * nb_symb)), symb)
            str_array.append(to_add)

    # Part of duration that is under a second
    symbols_list = ["ms", "μs"]
    nb_symb = int(used_duration)
    remaining_secs_nb = used_duration
    for symb in symbols_list:
        if remaining_secs_nb == 0:  # If the duration to format has become null, we finish the step early
            break
        remaining_secs_nb = 1000 * (remaining_secs_nb - nb_symb)
        nb_symb = int(remaining_secs_nb)
        if nb_symb > 0:
            to_add = "{:s}{:s}".format(str(int(the_sign * nb_symb)), symb)
            str_array.append(to_add)

    return " ".join(str_array)


def get_display_progress_function(logger, maximum_count=None, percentage_interval=5, 
                                  count_interval=1000, personalized_progress_message=None, 
                                  logging_level="debug"):
    logging_lvl = format_logging_level(logging_level)
    
    if maximum_count is not None and maximum_count > 1:
        used_percentage_interval = float(abs(percentage_interval))
        if used_percentage_interval == 0:
            used_percentage_interval = 1
        # k belongs to [0; maximum_count-1]; more interested in specifying that the progress has begun, that in specifying that it just ended.
        possible_multipliers = numpy.array([w for w in range(0, int(numpy.floor(100 / used_percentage_interval)) + 1)], dtype=numpy.int)
        threshold_count_values = numpy.array(- numpy.floor(-maximum_count * used_percentage_interval * possible_multipliers / 100), dtype=numpy.int)
        corresponding_percentage_values = used_percentage_interval * possible_multipliers
        total_number = maximum_count*numpy.ones(threshold_count_values.shape, dtype=numpy.int)
        a_dict = dict(zip(threshold_count_values, zip(corresponding_percentage_values, threshold_count_values, total_number)))
        
        l = len(str(maximum_count))
        message = "Current progress = {percentage:6.2f}% ({current:"+str(l)+"d} / {total:"+str(l)+"d})"
        if personalized_progress_message is not None:
            message = personalized_progress_message
        
        def display_message(k):
            if k in a_dict:
                a_tuple = a_dict[k]
                logger.log(logging_lvl, message.format(percentage=a_tuple[0], current=a_tuple[1], total=a_tuple[2]))
        result = display_message
    else:
        used_count_interval = count_interval
        if used_count_interval is None or used_count_interval < 1:
            used_count_interval = 1000
        used_count_interval = int(used_count_interval)
        
        message = "Iteration n°{current:d}"
        if personalized_progress_message is not None:
            message = personalized_progress_message
        
        result = lambda k: logger.log(logging_lvl, message.format(current=k)) if k % used_count_interval == 0 else None
    
    return result

def save_sparse_csr(file_path, sparse_matrix):
    """Save a scipy.sparse.csr_matrix as a '.npz' archive of '.npy' files.

    :param file_path: local path where the archive will be written
    :param sparse_matrix: the csr_matrix object to save
    :return: None

    ..seealso: load_sparse_csr()
    """
    if not isinstance(sparse_matrix, sparse.csr_matrix):
        error_message = "Error: the input is not a scipy.sparse.csr_matrix (type = {:s})."
        error_message = error_message.format(str(type(sparse_matrix)))
        raise ValueError(error_message)
    numpy.savez_compressed(file_path, data=sparse_matrix.data, indices=sparse_matrix.indices,
                           indptr=sparse_matrix.indptr, shape=sparse_matrix.shape)


def load_sparse_csr(file_path):
    """Load a scipy.sparse.csr_matrix from a '.npz' archive of '.npy' files.

    :param file_path: local path where the archive is located
    :return: a scipy.sparse.csr_matrix

    ..seealso: save_sparse_csr()
    """
    loader = numpy.load(file_path)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                             shape=loader['shape'])

def l2_normalize(an_array, rows=True, inplace=False):
    """
        For inplace to be used, an_array must be a sparse matrix whose type is 
        compatible with the axis along which the user wants to normalize 
        (ex: 'csr' for rows=True).
    """
    return normalize(an_array, norm="l2", axis=int(rows), inplace=inplace)

def normalize(matrix, norm="l2", axis=1, inplace=False):
    """Normalize the matrix-like input.

    :param matrix: a matrix-like object: scipy.sparse matrix or numpy.ndarray
    :param norm: 'l2' (default), 'l1' or 'max'
    :param axis: 1 (default) or 0; specify along which axis to compute the average values that will
    be used to carry out the normalization. Use axis=1 to normalize the rows, and axis=0 to
    normalize the columns.
    :param inplace: boolean (default=False), whether or not to perform inplace normalization. Note
    that if the sparse matrix's format is not compatible with the axis along which the
    normalization is asked (csr for axis=1, csc for axis=0), another one will be created, and
    'inplace' will become moot. Also, will not apply if the input's dtype is 'int' instead of a
    type compatible with division, such as 'float'.
    :result: a matrix-like object
    """
    if isinstance(matrix, numpy.ndarray) and len(matrix.shape) == 1:
        # sklearn's normalize function does not work in that case
        matrix = _check_numpy_array(matrix, copy=not inplace,
                                    dtype=(numpy.float64, numpy.float32, numpy.float16))

        if norm == "l2":
            denominator = numpy.sqrt(numpy.sum(numpy.power(matrix, 2)))
        elif norm == "l1":
            denominator = numpy.sum(numpy.abs(matrix))
        elif norm == "max":
            denominator = numpy.max(numpy.abs(matrix))
        else:
            raise NotImplementedError
        if denominator > 0:
            matrix /= denominator
        return matrix
    else:
        return sklearn.preprocessing.normalize(matrix, norm=norm, axis=axis, copy=not inplace)

def nan_equal(a,b):
    try:
        numpy.testing.assert_equal(a,b)
    except AssertionError:
        return False
    return True


def get_containing_folder_name(file_path):
    return os.path.basename(os.path.dirname(file_path))

def get_folder_name(folder_path):
    """Can you trust the input path to be a 'true' folder path?"""
    return os.path.basename(os.path.dirname(os.path.join(folder_path, "")))

def list_subdirectories(folder_path):
    """Returns list of name of folders at the root of the folder whose folder path is input."""
    return next(os.walk(folder_path))[1]

def list_files_root(folder_path):
    """Returns list of filepath of files at the root of the folder whose folder path is input."""
    return [f2 for f2 in [os.path.join(folder_path, f1) for f1 in os.listdir(folder_path)] if (os.path.isfile(f2))]

def check_and_create_folder(folder_path, need_write=True, anew=False, verbose=False):
    """
        Check that folder corresponding to input exists, and create it if needed.
         
        :type folder_path: string
        :param folder_path: path to the folder to check
        :type need_write: boolean
        :param need_write: ask (or not) that the check also accounts for write 
        permission
        :type anew: boolean
        :param anew: ask (or not) that a new folder be created if one already 
        exists
        :type verbose: boolean
        :param verbose: ask (or not) that a warning be displayed if the folder 
        does not exists and need_write = False
        :return: None
    """
    newly_created = False
    if os.path.exists(folder_path):
        if not os.path.isdir(folder_path):
            error_message = "There is an element located at '{:s} that is not a "\
            "folder.'"
            error_message = error_message.format(folder_path)
            raise ValueError(error_message)
        if need_write:
            if (not os.access(folder_path, os.W_OK)):
                error_message = "The process does not have write access to the "\
                "folder located at '{:s}.'"
                error_message = error_message.format(folder_path)
                raise PermissionError(error_message)
            if (anew):
                shutil.rmtree(folder_path)
                os.makedirs(folder_path)
    else:
        if (need_write):
            os.makedirs(folder_path)
            newly_created = True
        else:
            message = "The folder located at '{:s}' does not exist.\nSince it "\
            "was not expected to write in it ('need_write' = False), the program"\
            " will not attempt to create it."
            message = message.format(folder_path)
            if (verbose):
                logger.warning(message)
    return newly_created


####################
# Evaluation utils #
####################
class EvaluationResult(object):
    
    def __init__(self, file_name, total_questions_nb, file_vocab_size, embedding_common_vocab_size):
        """
        if total_questions_nb < 1:
            error_message = "An evaluation data file must contain at least one question "\
            "('total_questions_nb' = {:d})."
            error_message = error_message.format(total_questions_nb)
            raise ValueError(error_message)
        """
        """
        if file_vocab_size < 1:
            error_message = "An evaluation data file must contain at least one "\
            "word ('file_vocab_size' = {:d})."
            error_message = error_message.format(file_vocab_size)
            raise ValueError(error_message)
        """
        if embedding_common_vocab_size > file_vocab_size:
            error_message = "'embedding_common_vocab_size' ({:d}) cannot be "\
            "higher than 'file_vocab_size' ({:s})!"
            error_message = error_message.format(embedding_common_vocab_size, file_vocab_size)
            raise ValueError(error_message)
        self.file_name = file_name
        self.total_questions_nb = total_questions_nb
        self.file_vocab_size = file_vocab_size
        self.embedding_common_vocab_size = embedding_common_vocab_size
    
    def __eq__(self, another):
        val = (self.file_name == another.file_name
               and nan_equal(self.total_questions_nb, another.total_questions_nb)
               and nan_equal(self.file_vocab_size, another.file_vocab_size)
               and nan_equal(self.embedding_common_vocab_size, another.embedding_common_vocab_size))
        return val
    
    def __str__(self):
        a_string = "\n".join(["file_name: {:s}".format(str(self.file_name)),
                              "total_questions_nb: {:s}".format(str(self.total_questions_nb)),
                              "file_vocab_size: {:s}".format(str(self.file_vocab_size)),
                              "embedding_common_vocab_size: {:s}".format(str(self.embedding_common_vocab_size))])
        return a_string    
    
    @classmethod
    def check_consistency_of_results_lists(cls, results_lists_list):
        for i, results_list in enumerate(results_lists_list):
            if i == 0:
                files_data_list = [(r.file_name, r.total_questions_nb, r.file_vocab_size) for r in results_list]
            else:
                # Check that list size is the same
                if len(results_list) != len(files_data_list):
                    error_message = "size of list ID n°{:d} ({:d}) is different from size of first list ({:d}, ID n°{:d})"
                    error_message = error_message.format(i, len(results_list), len(files_data_list), 0)
                    raise ValueError(error_message)
                # Check that list is in the same order wrt to evaluation data files (and that the data associated to each of them is the same)
                for j, r in enumerate(results_list):
                    a_tuple = (r.file_name, r.total_questions_nb, r.file_vocab_size)
                    if files_data_list[j] != a_tuple:
                        error_message = "result ID n°{:d} of results list ID n°{:d} is different from result ID n°{:d} of the first results list ('{:s}' VS '{:s}')."
                        error_message = error_message.format(j, i, j, str(a_tuple), str(files_data_list[j]))
                        raise ValueError(error_message)
    
        return files_data_list
    
    @classmethod
    def _get_legend_string(cls, embeddings_collection):
        a_string = "    word embeddings n° {:d}: {:s}"
        f_string = "label = '{:s}'"
        identification_string =\
         "\n".join([a_string.format(i, f_string.format(e.label)) for i, e in enumerate(embeddings_collection)])
        return identification_string



class SimilarityResult(EvaluationResult):
    
    def __init__(self, file_name, total_questions_nb, file_vocab_size, embedding_common_vocab_size,
                 true_sim_values, computed_sim_values, drop_raw_data=True):
        super().__init__(file_name, total_questions_nb, file_vocab_size, embedding_common_vocab_size)
        
        answered_questions_nb = len(computed_sim_values)
        asked_questions_nb = len(true_sim_values)
        if asked_questions_nb != answered_questions_nb:
            error_message = "'answered_questions_nb' ({:d}) cannot be different "\
            "than 'asked_questions_nb' ({:s})!"
            error_message = error_message.format(answered_questions_nb, asked_questions_nb)
            raise ValueError(error_message)
        if answered_questions_nb > total_questions_nb:
            error_message = "'answered_questions_nb' ({:d}) cannot be "\
            "higher than 'total_questions_nb' ({:s})!"
            error_message = error_message.format(answered_questions_nb, total_questions_nb)
            raise ValueError(error_message)
        
        self.true_sim_values = true_sim_values
        self.computed_sim_values = computed_sim_values
        self.answered_questions_nb = answered_questions_nb
        
        self.pearson_tuple = None
        self.spearman_result = None
        self.post_result_computation = False
        
        self.drop_raw_data = drop_raw_data
        self.compute_stats = True
        if self.compute_stats:
            self.compute_results_stats(self.drop_raw_data)
    
    def compute_results_stats(self, drop_raw_data=True):
        if not self.post_result_computation:
            # Precision
            if self.answered_questions_nb > 0:
                # Pearson rank score
                pearson_tuple = scipy.stats.pearsonr(self.computed_sim_values, self.true_sim_values)
                # Spearman rank score
                spearman_result = scipy.stats.spearmanr(self.computed_sim_values, self.true_sim_values)
            else:
                pearson_tuple = (numpy.NaN, numpy.NaN)
                spearman_result = scipy.stats.stats.SpearmanrResult(numpy.NaN, numpy.NaN)
            # Questions coverage
            questions_coverage = float(self.answered_questions_nb) / float(self.total_questions_nb) if self.total_questions_nb > 0 else numpy.NaN
            # Vocabulary coverage
            vocabulary_coverage = float(self.embedding_common_vocab_size) / float(self.file_vocab_size) if self.file_vocab_size > 0 else numpy.NaN
            # Setting the computed values
            self.pearson_tuple = pearson_tuple
            self.spearman_result = spearman_result
            self.questions_coverage = questions_coverage
            self.vocabulary_coverage = vocabulary_coverage
            # Dropping the raw result so as to free some memory, potentially
            if drop_raw_data:
                self.computed_sim_values = None
                self.true_sim_values = None
            
            self.post_result_computation = True
        return self
            
    
    def format_results_stats(self, row=True):
        if not self.post_result_computation:
            self.compute_results_stats()
        if row:
            #formatted_string = "pea = {:+6.3f}%, sp = {:+6.3f} (qc = {:6.2f}%, vc = {:6.2f}%)"
            formatted_string = "{:+6.3f}, {:+6.3f} ({:6.2f}%, {:6.2f}%)"
        else:
            formatted_string = "\n".join(["pea = {:+6.3f}", "sp = {:+6.3f}", "qc = {:6.2f}%", "vc = {:6.2f}%"])
        formatted_string = formatted_string.format(self.pearson_tuple[0],
                                                   self.spearman_result[0],
                                                   100*self.questions_coverage,
                                                   100*self.vocabulary_coverage)
        return formatted_string
    
    def __eq__(self, another):
        val = super().__eq__(another)
        val = (val and nan_equal(self.answered_questions_nb, another.answered_questions_nb)
               and nan_equal(self.questions_coverage, another.questions_coverage)
               and nan_equal(self.vocabulary_coverage, another.vocabulary_coverage)
               and ((nan_equal(self.true_sim_values, another.true_sim_values) 
                     and nan_equal(self.computed_sim_values, another.computed_sim_values))
                     or (nan_equal(self.pearson_tuple, another.pearson_tuple) 
                         and nan_equal(self.spearman_result, another.spearman_result)))
               )
        return val
    
    def __str__(self):
        a_string = super().__str__()
        a_string = "\n".join([a_string,
                              "true_sim_values: {:s}".format(str(self.true_sim_values)),
                              "computed_sim_values: {:s}".format(str(self.computed_sim_values)),
                              "answered_questions_nb: {:s}".format(str(self.answered_questions_nb)),
                              "questions_coverage: {:s}".format(str(self.questions_coverage)),
                              "vocabulary_coverage: {:s}".format(str(self.vocabulary_coverage)),
                              "pearson_tuple: {:s}".format(str(self.pearson_tuple)),
                              "spearman_result: {:s}".format(str(self.spearman_result)),
                              "post_result_computation: {:s}".format(str(self.post_result_computation))])
        return a_string
    
    @classmethod
    def get_format_legend(self):
        #legend_string = "\n".join(["qnb <=> questions nb", "vs <=> vocabulary size", "pea <=> pearson's score", "sp <=> spearman's score", "qc <=> questions coverage", "vc <=> vocabulary coverage"])
        legend_string = "\n".join(["legend:", "    data infos: x, y <=> file questions total number, file vocabulary size", 
                                   "    word embedding stats: x, y (z%, a%) <=> pearson's score, spearman's score  (questions coverage, vocabulary coverage)"])
        return legend_string
    



class AnalogyResult(EvaluationResult):
    
    POSSIBLE_AGREGGATE_TYPES = set(["micro-average", "macro-average"])
    
    def __init__(self, file_name, total_questions_nb, file_vocab_size, embedding_common_vocab_size,
                 correct_answers, incorrect_answers, drop_raw_data=True):
        super().__init__(file_name, total_questions_nb, file_vocab_size, embedding_common_vocab_size)
        
        lc = len(correct_answers)
        li = len(incorrect_answers)
        if lc + li > total_questions_nb:
            error_message = "Sum of correct ({:d}) and incorrect ({:d}) answers "\
            "numbers (= {:d}) is strictly superior to total number of questions "\
            "in file ({:d}), not possible."
            error_message = error_message.format(lc, li, lc+li, total_questions_nb)
            raise ValueError(error_message)
        
        self.correct_answers = correct_answers
        self.incorrect_answers = incorrect_answers
        self.correct_answers_nb = len(self.correct_answers)
        self.answered_questions_nb = self.correct_answers_nb + len(self.incorrect_answers)
        
        self.precision = None
        self.questions_coverage = None
        self.vocabulary_coverage = None
        self.post_result_computation = False
        
        self.drop_raw_data = drop_raw_data
        self.compute_stats = True
        
        if self.compute_stats:
            self.compute_results_stats(self.drop_raw_data)
    
    def compute_results_stats(self, drop_raw_data=True):
        if not self.post_result_computation:
            # Precision
            if self.answered_questions_nb > 0:
                precision = float(self.correct_answers_nb) / float(self.answered_questions_nb) if self.answered_questions_nb > 0 else numpy.NaN
            else:
                precision = numpy.NaN
            # Questions coverage
            questions_coverage = float(self.answered_questions_nb) / float(self.total_questions_nb) if self.total_questions_nb > 0 else numpy.NaN
            # Vocabulary coverage
            vocabulary_coverage = float(self.embedding_common_vocab_size) / float(self.file_vocab_size) if self.file_vocab_size > 0 else numpy.NaN
            # Setting the computed values
            self.precision = precision
            self.questions_coverage = questions_coverage
            self.vocabulary_coverage = vocabulary_coverage
            # Dropping the raw result so as to free some memory, potentially
            if drop_raw_data:
                self.correct_answers = None
                self.incorrect_answers = None
            
            self.post_result_computation = True
        return self
    
    def format_results_stats(self):
        row=True
        if not self.post_result_computation:
            self.compute_results_stats()
        if row:
            #formatted_string = "pre = {:6.2f}% (qc = {:6.2f}%, vc = {:6.2f}%)"
            formatted_string = "{:6.2f}% ({:6.2f}%, {:6.2f}%)"
        else:
            formatted_string = "\n".join(["pre = {:6.2f}%", "qc  = {:6.2f}%", "vc  = {:6.2f}%"])
        formatted_string = formatted_string.format(100*self.precision,
                                                   100*self.questions_coverage,
                                                   100*self.vocabulary_coverage)
        return formatted_string
    
    def __eq__(self, another):
        val = super().__eq__(another)
        val = (val and nan_equal(self.answered_questions_nb, another.answered_questions_nb)
               and nan_equal(self.correct_answers_nb, another.correct_answers_nb)
               and nan_equal(self.questions_coverage, another.questions_coverage)
               and nan_equal(self.vocabulary_coverage, another.vocabulary_coverage)
               and ((nan_equal(self.correct_answers, another.correct_answers) 
                     and nan_equal(self.incorrect_answers, another.incorrect_answers)) 
                    or (nan_equal(self.precision, another.precision)))
               )
        return val
    
    def __str__(self):
        a_string = super().__str__()
        a_string = "\n".join([a_string,
                              "correct_answers: {:s}".format(str(self.correct_answers)),
                              "incorrect_answers: {:s}".format(str(self.incorrect_answers)),
                              "correct_answers_nb: {:s}".format(str(self.correct_answers_nb)),
                              "answered_questions_nb: {:s}".format(str(self.answered_questions_nb)),
                              "questions_coverage: {:s}".format(str(self.questions_coverage)),
                              "vocabulary_coverage: {:s}".format(str(self.vocabulary_coverage)),
                              "precision: {:s}".format(str(self.precision)),
                              "post_result_computation: {:s}".format(str(self.post_result_computation))])
        return a_string
    
    @classmethod
    def get_format_legend(self):
        #legend_string = "\n".join(["qnb <=> questions nb", "vs <=> vocabulary size", "pre <=> precision", "qc <=> questions coverage", "vc <=> vocabulary coverage"])
        legend_string = "\n".join(["legend:", "    data infos: x, y <=> file questions total number, file vocabulary size", 
                                   "    word embedding stats: x% (y%, z%) <=> precision (questions coverage, vocabulary coverage)"])
        return legend_string
    
    @classmethod
    def aggregate_results(cls, results_list, aggregated_field_name, aggregate_type="micro-average"):
        # Make sure stats are computed for everyone
        results_list = [r.compute_results_stats()for r in results_list ]
        
        template_name = "total {:s} ('{:s}')"
        
        # Then, compute the aggregate
        if aggregate_type == "micro-average":
            name = template_name.format(aggregate_type, aggregated_field_name)
            questions_nb = sum(r.total_questions_nb for r in results_list)
            answered_questions_nb = sum(r.answered_questions_nb for r in results_list)
            correctly_answered_question_nb = sum(r.correct_answers_nb for r in results_list)
            
            micro_average_prec = float(correctly_answered_question_nb) / float(answered_questions_nb) if answered_questions_nb > 0 else numpy.NaN
            micro_average_questions_coverage = float(answered_questions_nb) / float(questions_nb) if questions_nb > 0 else numpy.NaN
            micro_average_vocabulary_coverage = numpy.NaN
            micro_average_vocabulary_size = numpy.NaN
            
            aggregate_object = cls(name, questions_nb, micro_average_vocabulary_size, -1, [], [], False)
            aggregate_object.precision = micro_average_prec
            aggregate_object.questions_coverage = micro_average_questions_coverage
            aggregate_object.vocabulary_coverage = micro_average_vocabulary_coverage
            aggregate_object.post_result_computation = True
            
        elif(aggregate_type == "macro-average"):
            def nanaverage(a_list):
                an_array = numpy.array(a_list)
                to_average = an_array[~numpy.isnan(an_array)]
                value = numpy.nanmean(to_average) if len(to_average) > 0 else numpy.NaN
                return value
            name = template_name.format(aggregate_type, aggregated_field_name)
            questions_nb = int(numpy.around(numpy.nanmean([float(r.total_questions_nb) for r in results_list])))
            answered_questions_nb = int(numpy.around(numpy.nanmean([float(r.answered_questions_nb) for r in results_list])))
            correctly_answered_question_nb = int(numpy.around(numpy.nanmean([float(r.correct_answers_nb) for r in results_list])))
            
            macro_average_prec = nanaverage([r.precision for r in results_list])
            macro_average_questions_coverage = nanaverage([r.questions_coverage for r in results_list])
            macro_average_vocabulary_coverage = nanaverage([r.vocabulary_coverage for r in results_list])
            macro_average_vocabulary_size = int(numpy.around(nanaverage([r.file_vocab_size for r in results_list])))
            
            aggregate_object = cls(name, questions_nb, macro_average_vocabulary_size, -1, [], [], False)
            aggregate_object.precision = macro_average_prec
            aggregate_object.questions_coverage = macro_average_questions_coverage
            aggregate_object.vocabulary_coverage = macro_average_vocabulary_coverage
            aggregate_object.post_result_computation = True
        else:
            error_message = "Wrong aggregate aggregate_type, possible types are among '{:s}'."
            error_message = error_message.format(str(cls.POSSIBLE_AGREGGATE_TYPES))
            raise ValueError(error_message)
        
        return aggregate_object




####################################
# Common multiprocessing resources #
####################################
class TempEmbeddingCollectionManager(object):
    '''
    classdocs
    '''
    
    def __init__(self, embeddings_collection, root_temp_folder_path=None):
        self.embeddings_collection = embeddings_collection
        self.root_temp_folder_path = root_temp_folder_path
        self._logger = logging.getLogger("{}.{}".format(__name__, self.__class__.__name__))
        
    def __enter__(self):
        new_embeddings_dir_path_collection, temporary_folder_path =\
         self._create_temp_embedding_collection(self.embeddings_collection)
        self.temp_embeddings_dir_path_collection = new_embeddings_dir_path_collection
        self.temporary_folder_path = temporary_folder_path
        return self
    
    def __exit__(self, *args):
        # We remove the temporary embeddings collection folder
        shutil.rmtree(self.temporary_folder_path)
    
    def _create_temp_embedding_collection(self, embeddings_collection):
        logger = self._logger
        
        # Defining temp folder path where the embeddings will be stored
        temp_folder_path =\
         tempfile.mkdtemp(suffix=".emb_collec", dir=self.root_temp_folder_path)
        
        temp_embeddings_folder_path_collection = []
        
        with InputOutputLoggerTimer(logger, "debug", 
                                    "creating and saving a clone for each embedding(s) ({:d} embeddings)".format(len(embeddings_collection))):
            for i, embedding in enumerate(embeddings_collection):
                temp_embedding_folder_name =\
                 "{:s}{:d}".format(TEMPORARY_EMBEDDING_NAME_PREFIX, i)
                temp_embedding_folder_path =\
                 os.path.join(temp_folder_path, temp_embedding_folder_name)
                embedding_copy = embedding.copy()
                embedding_copy.save_in_folder(temp_embedding_folder_path)
                temp_embeddings_folder_path_collection.append(temp_embedding_folder_path)
        
        return temp_embeddings_folder_path_collection, temp_folder_path
    
    def get_temp_embedding_id_from_name(self, temp_embedding_name):
        pattern = "{:s}".format(TEMPORARY_EMBEDDING_NAME_PREFIX) + "(\d+)"
        return int(re.search(pattern, temp_embedding_name).group(1))

def _post_process_results_list_per_embedding(results):
    results.sort(key=lambda x: x.file_name)
    return results

def _compute_embedding_evaluation_one_folder(embedding, folder_path, evaluation_method_one_file, 
                                             workers_nb=1, **kwargs):
    results = []
    # List everything in the folder, only keep the files that ends with '.txt'
    files = list_files_root(folder_path)
    files = [filepath for filepath in files if (filepath.endswith(".txt"))]
    files_nb = len(files)
    
    if files_nb == 0:
        a_message = NO_FILE_WARNING_MESSAGE_TEMPLATE.format(folder_path)
        logger.warning(a_message)
        
    else:
        workers_nb = min(max(workers_nb, 1), files_nb)
        
        if workers_nb == 1:
            results = []
            for filepath in files:
                a_result = evaluation_method_one_file(embedding, filepath, **kwargs)
                results.append(a_result)
        else:
            def map_function(worker_id, input_queue, output_queue, exception_queue):
                try:
                    while True:
                        job = input_queue.get()
                        if job is not None:
                            file_path = job
                            a_result = evaluation_method_one_file(embedding, file_path, **kwargs)
                            output_queue.put(a_result)
                        else:
                            break
                    logger.debug("'Map' worker n°{:d} about to terminate.".format(worker_id))
                    output_queue.put(worker_id) # Signal that the worker can be terminated
                except Exception:
                    tb = traceback.format_exc()
                    exception_queue.put(tb)
                        
            def job_dispatcher_function(input_queue, files, workers_nb, exception_queue):
                try:
                    # Filling the queue with jobs
                    for filepath in files:
                        job = filepath
                        input_queue.put(job)
                    logger.debug("Finished inputing jobs into input_queue.")
    
                    # Signal termination
                    for _ in range(workers_nb):
                        input_queue.put(None)
                    
                    logger.debug("'Job dispatcher' worker about to terminate.")
                except Exception:
                    tb = traceback.format_exc()
                    exception_queue.put(tb)
            
            with InputOutputLoggerTimer(logger, "debug", "multiprocessing (using {:d} process).".format(workers_nb)):
                #  Creating the queues
                input_queue = Queue(maxsize=0)
                output_queue = Queue(maxsize=0)
                exception_queue = Queue(maxsize=0)
                
                # Creating the mapping workers
                workers = []
                for worker_id in range(workers_nb):
                    a_worker = Process(target=map_function, args=(worker_id, input_queue, output_queue, exception_queue))
                    a_worker.daemon = True
                    workers.append(a_worker)
                
                # Creating the job dispatching worker
                job_dispatcher_worker = Process(target=job_dispatcher_function, args=(input_queue, files, len(workers), exception_queue))
                job_dispatcher_worker.daemon = True 
                
                # Starting the workers
                for a_worker in workers:
                    a_worker.start()
                job_dispatcher_worker.start() # Job dispatching starts now
                
                # Function to call if an exception is encountered
                def _manage_in_case_of_exception(workers):
                    for worker in workers:
                        worker.terminate()
                    job_dispatcher_worker.terminate()
                
                # Receiving the results
                try:
                    results = []
                    stopped_workers_nb = 0
                    while True:
                        # First, check if one worker has encountered an Exception
                        try:
                            a_traceback = exception_queue.get_nowait()
                            _manage_in_case_of_exception(workers)
                            raise Exception(a_traceback)
                        except multiprocessing.queues.Empty:
                            pass
                        
                        # If no one has, listen during one second for regular outputs
                        try:
                            job = output_queue.get(True, 1)
                            if job is not None:
                                if isinstance(job, int):
                                    workers[job].join()
                                    stopped_workers_nb += 1
                                    logger.debug("'Map' worker n°{:d} terminated, {:d} remaining.".format(job, workers_nb-stopped_workers_nb))
                                else:
                                    a_result = job
                                    results.append(a_result)
                        except multiprocessing.queues.Empty:
                            pass
                        
                        if stopped_workers_nb == workers_nb:
                            break
                except Exception:
                    _manage_in_case_of_exception(workers)
                    raise
                
                # Wait for workers to terminate
                for w in workers:
                    w.join()
                logger.debug("'Map' workers all terminated.")
                job_dispatcher_worker.join()
                logger.debug("'Job dispatcher' worker terminated.")
        
        _post_process_results_list_per_embedding(results)
    
    return results


def _compute_evaluation_folders_multiproc(embeddings_collection, folder_path,
                                          evaluation_method_one_file, workers_nb=1,
                                          subfolder=False,
                                          temp_folder_path=None,
                                          **kwargs):
    """
    Evaluates embedding on similarities dataset
    :param embeddings_collection: the list of embeddings to evaluate
    :param folder_path: a filepath or a dir path to an evaluation dataset
    :return: list of list of evaluation results, or dict of list of list of evaluation results
    """
    empty_list = [[] for _ in embeddings_collection]
    
    workers_nb = max(min(workers_nb, cpu_count()), 1)
    
    # List everything in the folder (subfolder(s)), and keep only the files that ends with '.txt'
    directory_name_list = []
    if not subfolder:
        files = sorted(list_files_root(folder_path))
        current_folder_name = get_folder_name(folder_path)
        directory_name_list.append(current_folder_name)
        root_folder_path = os.path.dirname(os.path.dirname(os.path.join(folder_path, "")))
        logging_message = "evaluating with data from folder located at '{:s}'".format(folder_path)
    else:
        files = []
        subdirectories_name_list = sorted(list_subdirectories(folder_path))
        for current_folder_name in subdirectories_name_list:
            current_folder_path = os.path.join(folder_path, current_folder_name)
            files += sorted(list_files_root(current_folder_path))
        directory_name_list.extend(subdirectories_name_list)
        root_folder_path = folder_path
        logging_message = "evaluating with data from subfolders of folder located at '{:s}'".format(folder_path)

    files = [file_path for file_path in files if (file_path.endswith(".txt"))]
    files_nb = len(files)
    embedding_nb = len(embeddings_collection)
    
    # Then... # <= This is a very useful comment
    with InputOutputLoggerTimer(logger, "info", logging_message):
        if embedding_nb == 0:
            warning_message = "The list of embedding to evaluate is empty'"
            logger.warning(warning_message)
            results = dict((name, []) for name in directory_name_list)
        
        elif files_nb == 0:
            warning_message = "No evaluation data files in folder '{:s}' (or its subfolders if 'subfolder = True')."
            warning_message.format(folder_path)
            logger.warning(warning_message)
            results = dict((name, empty_list) for name in directory_name_list)
        
        else:
            results = dict((name, empty_list) for name in directory_name_list)
            
            # No need to take the big gun if the user asked for no multiprocessing, or if there is only one embedding to evaluate
            if workers_nb == 1 or embedding_nb == 1:
                for current_folder_name in directory_name_list:
                    results_lists_list = []
                    current_folder_path = os.path.join(root_folder_path, current_folder_name)
                    for embedding in embeddings_collection:
                        results_list = _compute_embedding_evaluation_one_folder(embedding,
                                        current_folder_path, evaluation_method_one_file,
                                        workers_nb=workers_nb, **kwargs)
                        results_lists_list.append(results_list)
                    results[current_folder_name] = results_lists_list
            
            else:
                
                # Create temporary versions of the embeddings
                with TempEmbeddingCollectionManager(embeddings_collection, root_temp_folder_path=temp_folder_path) as tecm:
                    temp_embeddings_dir_path_collection = tecm.temp_embeddings_dir_path_collection
                    
                    # Check that no problem occurred with temporary embeddings creation 
                    new_embedding_nb = len(temp_embeddings_dir_path_collection)
                    if new_embedding_nb != embedding_nb:
                        error_message = "Inconsistency between length of new collection ({:d})"\
                        " and length of original one ({:d})."
                        error_message = error_message.format(new_embedding_nb, embedding_nb)
                        raise ValueError(error_message)
                    
                    # Adjust the number of real workers to use if necessary
                    workers_nb = min(workers_nb, embedding_nb * files_nb)
                    
                    # Create list of jobs per embedding
                    #####
                    # WARNING: we assume we can store the whole lists of jobs at once in every queues! That may not be the case if the number of files and / or of embeddings is very high! 
                    #####
                    def create_queue(items_list):
                            queue = Queue(maxsize=0)
                            for worker_id in items_list:
                                queue.put(worker_id)
                            return queue
                    with InputOutputLoggerTimer(logger, "debug", 
                                                "defining jobs ({:d} embedding(s), {:d} file(s))".format(embedding_nb, files_nb)):
                        jobs_per_embedding = dict((dir_path, create_queue([(dir_path, file_path) for file_path in files])) for dir_path in temp_embeddings_dir_path_collection)
                    total_jobs_nb = len(temp_embeddings_dir_path_collection) * len(files)
                    percentage = 5
                    display_function =\
                     get_display_progress_function(logger, total_jobs_nb, percentage,
                                                   personalized_progress_message="Dispatched jobs progress = {percentage:.2f}% ({current:d} / {total:d})",
                                                   logging_level="debug")
                    
                    # Define the target function
                    def target_function(worker_id, input_queue, output_queue, exception_queue):
                        try:
                            previous_job_embedding_dir_path = None
                            while True:
                                job = input_queue.get()
                                if job is not None:
                                    job_embedding_dir_path = job[0]
                                    dataset_file_path = job[1]
                                    if (previous_job_embedding_dir_path is None
                                         or previous_job_embedding_dir_path != job_embedding_dir_path):
                                        embedding = Embedding.load_from_folder(job_embedding_dir_path)
                                        previous_job_embedding_dir_path = job_embedding_dir_path
                                    # Compute the result
                                    embedding_name = os.path.basename(job_embedding_dir_path)
                                    dataset_file_name = os.path.basename(dataset_file_path)
                                    job_string = "job (worker n°{:d}, '{:s}', '{:s}')".format(worker_id, embedding_name,dataset_file_name)
                                    with InputOutputLoggerTimer(logger, "debug", job_string, 
                                                                "Duration {:s} = {:s}.".format(job_string, "{:s}")):
                                        a_result = evaluation_method_one_file(embedding, dataset_file_path, **kwargs)
                                    # Output the result, and thus tell the main process that we want a job
                                    output = (os.path.basename(previous_job_embedding_dir_path), dataset_file_path, a_result)
                                    output_queue.put((worker_id, output))
                                else:
                                    break
                            logger.debug("Worker n°{:d} about to terminate.".format(worker_id))
                        except Exception:
                            tb = traceback.format_exc()
                            exception_queue.put(tb)
                    
                    # Carry out multiprocessing work
                    with InputOutputLoggerTimer(logger, "debug", 
                                                "multiprocessing (using {:d} process)".format(workers_nb)):
                        # Define the queues and communication devices for the workers tasked with the evaluation jobs
                        workers_input_queues = dict(enumerate(Queue(maxsize=0) for worker_id in range(workers_nb)))
                        output_queue = Queue(maxsize=0)
                        exception_queue = Queue(maxsize=0)
                        
                        # Create the workers
                        workers = []
                        for worker_id in range(workers_nb):
                            args = (worker_id, workers_input_queues[worker_id], output_queue, exception_queue)
                            a_worker = Process(target=target_function, args=args)
                            a_worker.daemon = True
                            workers.append(a_worker)
                        
                        # Fill the queues with jobs
                        logger.debug("{:d} jobs to carry out.".format(total_jobs_nb))
                        with InputOutputLoggerTimer(logger, "debug", "dispatching jobs to workers and carrying out jobs"):
                            # Define the respective first embedding loaded within each worker
                            workers_respective_previous_embedding = dict((worker_id, None) for worker_id in range(workers_nb))
                            
                            # Define function used to choose which job to give to a worker
                            def choose_job_to_dispatch(worker_id):
                                job = None
                                # Find the identity of the last embedding it evaluated
                                previous_embedding_dir_path = workers_respective_previous_embedding[worker_id]
                                previous_embedding_queue = jobs_per_embedding[previous_embedding_dir_path]
                                # Check if there are jobs left to carry out for this embedding
                                nb_remaining_jobs = previous_embedding_queue.qsize()
                                if nb_remaining_jobs > 0:
                                    job = previous_embedding_queue.get()
                                else:
                                    # Then we must choose a job related to another embedding:
                                    # we first choose the embedding for which there is only one job remaining, or the one with the maximum number of job remaining
                                    remaining_embeddings_dir_path_with_jobs =\
                                     [dir_path for dir_path in jobs_per_embedding.keys() if jobs_per_embedding[dir_path].qsize() > 0]
                                    arg_sorted = sorted(remaining_embeddings_dir_path_with_jobs, key=lambda x: jobs_per_embedding[x].qsize())
                                    if len(arg_sorted) > 0:
                                        queue_arg_with_minimum_value = jobs_per_embedding[arg_sorted[0]]
                                        if queue_arg_with_minimum_value.qsize() == 1:
                                            job = queue_arg_with_minimum_value.get()
                                        else:
                                            job = jobs_per_embedding[arg_sorted[-1]].get()
                                return job
                            
                            # Start the workers
                            for a_worker in workers:
                                a_worker.start()
                            
                            # Function to call if an exception is encountered
                            def _manage_in_case_of_exception(workers):
                                for worker in workers:
                                    worker.terminate()
                            
                            # Finally, launch the work by starting the dispatching of the jobs
                            try:
                                outputs = []
                                dispatched_jobs_nb = 0
                                stopped_workers_nb = 0
                                # Initial job distribution: we previously made sure that there is enough jobs to cover this initial phase when we set 'workers_nb = min(...)
                                for worker_id in range(workers_nb):
                                    embedding_dir_path = temp_embeddings_dir_path_collection[worker_id % embedding_nb]
                                    job = jobs_per_embedding[embedding_dir_path].get()
                                    workers_respective_previous_embedding[worker_id] = job[0]
                                    workers_input_queues[worker_id].put(job)
                                    dispatched_jobs_nb += 1
                                display_function(dispatched_jobs_nb)
                                # After dispatching the first jobs, wait for outputs, and give back a job if at least one remains
                                while True:
                                    # First, check if one worker has encountered an Exception
                                    try:
                                        a_traceback = exception_queue.get_nowait()
                                        _manage_in_case_of_exception(workers)
                                        raise Exception(a_traceback)
                                    except multiprocessing.queues.Empty:
                                        pass
                                    
                                    # If no one has, listen during one second for regular outputs
                                    try:
                                        output = output_queue.get(True, 1)
                                        worker_id = output[0]
                                        sub_output = output[1]
                                        outputs.append(sub_output)
                                        new_job = choose_job_to_dispatch(worker_id)
                                        workers_input_queues[worker_id].put(new_job)
                                        if new_job is not None:
                                            workers_respective_previous_embedding[worker_id] = new_job[0]
                                            dispatched_jobs_nb += 1
                                            display_function(dispatched_jobs_nb)
                                        else:
                                            stopped_workers_nb += 1
                                            logger.debug("'Map' worker n°{:d} asked to terminate, {:d} remaining.".format(worker_id, workers_nb-stopped_workers_nb))
                                    except multiprocessing.queues.Empty:
                                        pass
                                    
                                    if stopped_workers_nb == workers_nb:
                                        break
                                logger.debug("All {:d} 'map' worker terminated.".format(workers_nb))
                            except Exception:
                                _manage_in_case_of_exception(workers)
                                raise
                              
                        # If we came to this stage, that means that each worker has terminated / is about to terminate without problem, so we can join them, and proceed
                        for worker in workers:
                            worker.join()
                
                # Now, just re-arrange the results:
                # For each folder, check the consistency, and give the list of each embedding's list of results
                # Check bijection between list of outputs, and list of cartesian product "embeddings * files"
                expected_set = set(itertools.product((os.path.basename(e_path) for e_path in temp_embeddings_dir_path_collection), files))
                actual_set = set((e_name, f_path) for e_name, f_path, _ in outputs)
                diff = expected_set.difference(actual_set)
                if len(diff) > 0:
                    desc_string = "\n\t".join("Embedding n°{:d} on file '{:s}'".format(tecm.get_temp_embedding_id_from_name(e), f_path) for e, f_path in diff)
                    error_message = "Problem: the following evaluation were not performed:\n\t{:s}"
                    error_message = error_message.format(desc_string)
                    raise ValueError(error_message)
                
                # Then, order per folder
                outputs_per_folder = defaultdict(list)
                for output in outputs:
                    file_path = output[1]
                    outputs_per_folder[get_containing_folder_name(file_path)].append(output)
                
                # Then, for each folder, order per embedding and per files
                embeddings_results_lists_list_per_folder = {}
                for current_folder_name, outputs in outputs_per_folder.items():
                    embeddings_results_lists_list = []
                    temp_dict = defaultdict(list)
                    for output in outputs:
                        embedding_name = output[0]
                        result = output[2]
                        temp_dict[embedding_name].append(result)
                    for embedding_name, results_list in sorted(temp_dict.items(), key=lambda x: tecm.get_temp_embedding_id_from_name(x[0])):
                        embeddings_results_lists_list.append(_post_process_results_list_per_embedding(results_list))
                    embeddings_results_lists_list_per_folder[current_folder_name] = embeddings_results_lists_list
                
                # Finally, give desired output
                results.update(embeddings_results_lists_list_per_folder)
        
    return results





##############
# Similarity #
############## 
def format_similarity_results_table(similarity_results_lists_list): #row=True
    """
        What is important is to enforce the consistency of column width.
        Do not forget to add a column giving informations regarding each file.
        We get a list of results for each embedding, a list which will be displayed as a column.
        We get a list of list.
        
        First, check consistency of the lists:
            - same data files
            - in the same order
            - with the same data (questions nb, vocabulary size)
        
        If for Analogy, then, for each list, create the aggregated data objects
        
        Then, the final string must be created row by row:
            - create string numpy array, fill it with the cell string of each data row
            - for each column, define column width from max of cell string and size of column header
            - define the header row
            - create final string from header row and data rows
    """
    final_string = ""
    if len(similarity_results_lists_list) > 0:
        
        # First, checking consistency of the lists
        files_data_list = SimilarityResult.check_consistency_of_results_lists(similarity_results_lists_list)
        l = len(files_data_list)
        
        #if len(files_data_list) > 0:
        # Then, create string numpy array, with the header row
        max_qnb_size = numpy.max([len(str(r[1])) for r in files_data_list]) if l > 0 else numpy.NaN
        max_vs_size = numpy.max([len(str(r[2])) for r in files_data_list]) if l > 0 else 0
        
        embeddings_nb = len(similarity_results_lists_list)
        files_nb = len(files_data_list)
        string_array = numpy.empty(shape=(files_nb+1, embeddings_nb+2), dtype=object)
        string_array[:,0] = ["similarity filename(s)"] + [r[0] for r in files_data_list]
        string_array[:,1] = ["data infos"] + [("{:>"+str(max_qnb_size)+".0f}, {:>"+str(max_vs_size)+".0f}").format(numpy.float(r[1]), numpy.float(r[2])) for r in files_data_list]
        string_array[:,2:] = numpy.array([["word embeddings n°{:d}".format(i)]+[r.format_results_stats() for r in sr_list] for i, sr_list in enumerate(similarity_results_lists_list)]).T
        
        # Infer column width and create formatting string for each row
        colums_width = [max(len(s) for s in string_array[:,i]) for i in range(string_array.shape[1])]
        formatting_string_array = numpy.empty(shape=string_array.shape, dtype=object)
        for i in range(string_array.shape[0]):
            if i == 0:
                formatting_string_array[i,:] = ["{:"+"{:d}".format(cs)+"s}" for cs in colums_width]
            else:
                formatting_string_array[i,:] = ["{:"+"{:d}".format(colums_width[0])+"s}"]+["{:>"+"{:d}".format(cs)+"s}" for cs in colums_width[1:]]
        
        # Format the string array
        for i in range(string_array.shape[0]):
            for j in range(string_array.shape[1]):
                string_array[i,j] = formatting_string_array[i,j].format(string_array[i,j])
        
        # Finally, add the walls of the displayed table, and create the final string
        string_header_line = "| {:s} |".format(" | ".join(string_array[0,:]))
        dash_line = "".join("-" for _ in range(len(string_header_line)))
        string_list = [dash_line, string_header_line, dash_line]
        if l == 0:
            empty_line = "|{:s}|".format(("{:^"+str(len(string_header_line)-2)+"}").format("NaN"))
            string_list.append(empty_line)
        string_list.extend("| {:s} |".format(" | ".join(string_array[i,:])) for i in range(1,string_array.shape[0]))
        string_list.append(dash_line)
        final_string = "\n".join(string_list)
        
    return final_string

def _get_similarity_pre_report(similarity_evaluation_string):
    """
        Add information regarding hyperparameters and how to read the table, to an existing analogy evaluation string.
    """
    if similarity_evaluation_string != "":
        s_legend_string = SimilarityResult.get_format_legend()
        similarity_evaluation_string = "\n".join([s_legend_string, similarity_evaluation_string])
    return similarity_evaluation_string

def _get_final_similarity_report(embeddings_legend, similarity_pre_report):
    """
        Create analogy evaluation report from embeddings legend string and analogy pre report.
    """
    strings_list = [SIMILARITY_REPORT_TITLE, "Word embeddings' legend:", embeddings_legend, " ", similarity_pre_report]
    final_report = "\n".join(strings_list)
    return final_report



def _compute_similarity_one_file(embedding, filepath, drop_raw_data=True):
    # compute_results_stats=False: We give ourselves the possibility to use the results in a finer way than just already computing the stats and discarding the data
    word_index = embedding.vocabulary.word_index
    embedding_matrix = embedding.matrix
    nb_of_questions_in_file = 0
    similarity_words = {}
    embedding_common_vocab_size = numpy.NaN
    file_vocabulary = set()
    file_vocab_size = 0
    scores = []  # scores obtained with our embedding
    gold = []  # references scores
    line_nb = -1
    # filter words appearing in our embeddings and in similarity words file
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_nb, line in enumerate(f):
                # Preprocessing and spliting the line
                a_tuple = line.strip().lower().split("\t")
                try:
                    word1, word2, coefficient = a_tuple
                except Exception as e:
                    error_message =\
                     "Incorrect analogy question (line_nb = {:d}, line = '{:s}')".format(line_nb, line)
                    raise ValueError(error_message)
                nb_of_questions_in_file += 1
                # Preprocessing the words
                word1 = word1.strip()
                word2 = word2.strip()
                # Adding words to file vocabulary
                file_vocabulary.add(word1)
                file_vocabulary.add(word2)
                if word1 in word_index and word2 in word_index:
                    similarity_words[(word1, word2)] = float(coefficient)
    except OSError as e:
        raise e
    
    if line_nb == -1:
        a_message =\
         "The file located at '{:s}' is empty, cannot ask analogy questions.".format(filepath)
        logger.warning(a_message)
    else:
        to_answer_questions_nb = len(similarity_words)
        file_vocab_size = len(file_vocabulary)
        embedding_common_vocab_size = len(file_vocabulary.intersection(set(word_index.keys())))
        
        if to_answer_questions_nb == 0:
            a_message = "Vocabulary not rich enough for embedding (embedding's "\
            "label = {:s}), cannot ask a single question among the possible "\
            "ones in '{:s}'."
            a_message = a_message.format(embedding.label, filepath)
            logger.debug(a_message)
        else:
            
            if isinstance(embedding_matrix, numpy.ndarray):
                for (word1, word2), coefficient in similarity_words.items():
                    # TODO : à valider
                    word1_index = word_index[word1]
                    word2_index = word_index[word2]
                    ab = embedding_matrix[word1_index].dot(embedding_matrix[word2_index])
                    a2 = embedding_matrix[word1_index].dot(embedding_matrix[word1_index])
                    b2 = embedding_matrix[word2_index].dot(embedding_matrix[word2_index])
                    scores.append(ab / np.sqrt(a2 * b2))
                    gold.append(coefficient)
                    '''
                    logger.debug(
                        "word_1 = {}, word_2 = {}, score = {}, gold = {}".format(word1, word2, ab / np.sqrt(a2 * b2),
                                                                                 coefficient))
                    '''
            elif isinstance(embedding_matrix, sparse.csr_matrix):
                for (word1, word2), coefficient in similarity_words.items():
                    word1_index = word_index[word1]
                    word2_index = word_index[word2]
                    ab = (embedding_matrix[word1_index].dot(embedding_matrix[word2_index].T))[0,0]
                    a2 = (embedding_matrix[word1_index].dot(embedding_matrix[word1_index].T))[0,0]
                    b2 = (embedding_matrix[word2_index].dot(embedding_matrix[word2_index].T))[0,0]
                    scores.append(ab / np.sqrt(a2 * b2))
                    gold.append(coefficient)
            else:
                error_message = "The embedding matrix is neither a numpy.ndarray "\
                "nor a scipy.sparse.csr_matrix (current type = {:s})."
                error_message = error_message.format(str(type(embedding_matrix)))
                raise ValueError(error_message)
            
    file_name = os.path.basename(filepath)
    a_result = SimilarityResult(file_name, nb_of_questions_in_file,
                                file_vocab_size, embedding_common_vocab_size,
                                gold, scores, drop_raw_data=drop_raw_data)
    
    return a_result


def _similarity_file_evaluation(embeddings_collection, similarity_dataset_file_path, drop_raw_data=True):
    """
        Carry out similarity evaluation on the file located at the input file path.
        Returns:
            - a string containing the table of results to be displayed
            - the corresponding embeddings' legend
            - the list of the respective embeddings' results / performance on this file
    """
    with InputOutputLoggerTimer(logger, "info", 
                                "evaluating similarity with data from the file located at '{:s}'".format(similarity_dataset_file_path)):
        # Compute the results
        similarity_results_lists_list = []
        for embedding in embeddings_collection:
            a_similarity_result =_compute_similarity_one_file(embedding,
                                    similarity_dataset_file_path,
                                    drop_raw_data=drop_raw_data)
            similarity_results_lists_list.append([a_similarity_result])
        
        # Format the result into a report string
        similarity_evaluation_string = format_similarity_results_table(similarity_results_lists_list)
        # Get the embeddings legend string, at the same spot where the evaluation string is generated
        legend_string = SimilarityResult._get_legend_string(embeddings_collection)
        
    return similarity_evaluation_string, legend_string, a_similarity_result

def similarity_file_evaluation(embeddings_collection, similarity_dataset_file_path,
                               drop_raw_data=True, output_results=False):
    """
        Carry out similarity evaluation on the file located at the input file path.
        Return a report detailing the similarity performance of each embedding on this file.
    """
    similarity_evaluation_string, legend_string, a_similarity_result =\
     _similarity_file_evaluation(embeddings_collection,
            similarity_dataset_file_path, drop_raw_data=drop_raw_data)
    
    similarity_pre_report = _get_similarity_pre_report(similarity_evaluation_string)
    similarity_report = _get_final_similarity_report(legend_string, similarity_pre_report)
    
    if output_results:
        return similarity_report, a_similarity_result
    else:
        return similarity_report


def _similarity_one_folder_evaluation(embeddings_collection,
                                 similarity_dataset_folder_path,
                                 drop_raw_data=True,
                                 workers_nb=1,
                                 temp_folder_path=None):
    """
        Carry out similarity evaluation on all files contained within the folder located at the input folder path.
        Returns:
            - a string containing the table of results to be displayed
            - the corresponding embeddings' legend
            - the list of the respective embeddings' results / performance lists on this folder.
    """
    with InputOutputLoggerTimer(logger, "info", 
                                "evaluating similarity with data from the folder located at '{:s}'".format(similarity_dataset_folder_path)):
        dataset_folder_path = similarity_dataset_folder_path
        # Get results
        similarity_results_lists_list_dict =\
         _compute_evaluation_folders_multiproc(embeddings_collection,
            dataset_folder_path, _compute_similarity_one_file,
            drop_raw_data=drop_raw_data,
            workers_nb=workers_nb, subfolder=False,
            temp_folder_path=temp_folder_path)
        similarity_results_lists_list = list(similarity_results_lists_list_dict.values())[0]
        
        # Format results
        similarity_evaluation_string = format_similarity_results_table(similarity_results_lists_list)
        # Get the embeddings legend string, at the same spot where the evaluation string is generated
        legend_string = SimilarityResult._get_legend_string(embeddings_collection)
    
    return similarity_evaluation_string, legend_string, similarity_results_lists_list

def similarity_folder_evaluation(embeddings_collection,
                                 similarity_dataset_folder_path,
                                 drop_raw_data=True,
                                 workers_nb=1,
                                 temp_folder_path=None,
                                 output_results=False):
    """
        Carry out similarity evaluation on all files contained within the folder located at the input folder path.
        Return a report detailing the performance on each files.
    """
    # Carry out computation, and get evaluation string
    similarity_evaluation_string, legend_string, similarity_results_lists_list =\
     _similarity_one_folder_evaluation(embeddings_collection,
                                 similarity_dataset_folder_path,
                                 drop_raw_data=drop_raw_data,
                                 workers_nb=workers_nb,
                                 temp_folder_path=temp_folder_path)
    
    # Format the report from the evaluation string and the embedding's legend
    similarity_pre_report = _get_similarity_pre_report(similarity_evaluation_string)
    similarity_report = _get_final_similarity_report(legend_string, similarity_pre_report)
    
    # Outputs
    if output_results:
        return similarity_report, similarity_results_lists_list
    else:
        return similarity_report

def _similarity_evaluation(embeddings_collection, language="en",
                           similarity_dataset_folder_path=None,
                           drop_raw_data=True , workers_nb=1,
                           temp_folder_path=None):
    """
        Carry out similarity evaluation on all files contained within the folder located at the input folder path.
        Returns:
            - a string containing the table of results to be displayed
            - the corresponding embeddings' legend
            - the dict of each subfolder's list of the respective embeddings' results / performance lists.
    """
    # Get location of datasets to use 
    if similarity_dataset_folder_path is None:
        datasets_folder_path = _check_language_folder(language)
        similarity_dataset_folder_path =\
         os.path.join(datasets_folder_path, "similarity")
    
    # Get results
    with InputOutputLoggerTimer(logger, "info", 
                                "carrying out similarity evaluation for the embeddings collection ({:d} embeddings)".format(len(embeddings_collection))):
        similarity_results_lists_list_dict =\
             _compute_evaluation_folders_multiproc(embeddings_collection,
                similarity_dataset_folder_path, _compute_similarity_one_file,
                drop_raw_data=drop_raw_data,
                workers_nb=workers_nb, subfolder=False,
                temp_folder_path=temp_folder_path)
    
    # Format results
    similarity_results_lists_list = list(similarity_results_lists_list_dict.values())[0]
    similarity_evaluation_string = format_similarity_results_table(similarity_results_lists_list)
    # Get the embeddings legend string, at the same spot where the analogy evaluation string is generated
    legend_string = SimilarityResult._get_legend_string(embeddings_collection)
    
    return similarity_evaluation_string, legend_string, similarity_results_lists_list_dict


def similarity_evaluation(embeddings_collection, language="en",
                          similarity_dataset_folder_path=None,
                          drop_raw_data=True, workers_nb=1, temp_folder_path=None,
                          output_results=False):
    """
        doc
    """
    # Carry out computation, and get evaluation string
    similarity_evaluation_string, legend_string, similarity_results_lists_dict =\
     _similarity_evaluation(embeddings_collection, language, similarity_dataset_folder_path,
                            drop_raw_data=drop_raw_data, workers_nb=workers_nb,
                            temp_folder_path=temp_folder_path)
    
    # Format the report from the evaluation string and the embedding's legend
    similarity_pre_report = _get_similarity_pre_report(similarity_evaluation_string)
    similarity_report = _get_final_similarity_report(legend_string, similarity_pre_report)
    
    # Outputs
    if output_results:
        return similarity_report, similarity_results_lists_dict
    else:
        return similarity_report







###########
# Analogy #
###########
def format_analogy_results_table(analogy_results_lists_list, field_name=""): #row=True
    """
        What is important is to enforce the consistency of column width.
        Do not forget to add a column giving informations regarding each file.
        We get a list of results for each embedding, a list which will be displayed as a column.
        We get a list of list.
        
        First, check consistency of the lists:
            - same data files
            - in the same order
            - with the same data (questions nb, vocabulary size)
        
        If for Analogy, then, for each list, create the aggregated data objects
        
        Then, the final string must be created row by row:
            - create string numpy array, fill it with the cell string of each data row
            - for each column, define column width from max of cell string and size of column header
            - define the header row
            - create final string from header row and data rows
    """
    final_string = ""
    if len(analogy_results_lists_list) > 0:
        
        # First, checking consistency of the lists
        files_data_list = AnalogyResult.check_consistency_of_results_lists(analogy_results_lists_list)
        l = len(files_data_list)
        
        agg_present = False
        if len(files_data_list) > 1:
            # Then, add aggregated results to each list
            for AnalogyResults_list in analogy_results_lists_list:
                micro_average_analogy_result = AnalogyResult.aggregate_results(AnalogyResults_list, field_name, aggregate_type="micro-average")
                macro_average_analogy_result = AnalogyResult.aggregate_results(AnalogyResults_list, field_name, aggregate_type="macro-average")
                AnalogyResults_list.append(micro_average_analogy_result)
                AnalogyResults_list.append(macro_average_analogy_result)
            files_data_list.extend([(r.file_name, r.total_questions_nb, r.file_vocab_size) for r in [micro_average_analogy_result, macro_average_analogy_result]])
            agg_present = True
        
        # Then, create string numpy array, with the header row
        # Get max size of numbers 'qnb' and 'vs'
        max_qnb_size = numpy.max([len(str(r[1])) for r in files_data_list]) if l > 0 else numpy.NaN
        max_vs_size = numpy.max([len(str(r[2])) for r in files_data_list]) if l > 0 else numpy.NaN
        
        embeddings_nb = len(analogy_results_lists_list)
        files_nb = len(files_data_list)
        string_array = numpy.empty(shape=(files_nb+1, embeddings_nb+2), dtype=object)
        string_array[:,0] = ["'{:s}' analogy filename(s)".format(field_name)] + [r[0] for r in files_data_list]
        string_array[:,1] = ["data infos"] + [("{:>"+str(max_qnb_size)+".0f}, {:>"+str(max_vs_size)+".0f}").format(numpy.float(r[1]), numpy.float(r[2])) for r in files_data_list]
        string_array[:,2:] = numpy.array([["word embeddings n°{:d}".format(i)]+[r.format_results_stats() for r in sr_list] for i, sr_list in enumerate(analogy_results_lists_list)]).T
        
        # Infer column width and create formatting string for each row
        colums_width = [max(len(s) for s in string_array[:,i]) for i in range(string_array.shape[1])]
        formatting_string_array = numpy.empty(shape=string_array.shape, dtype=object)
        for i in range(string_array.shape[0]):
            if i == 0:
                formatting_string_array[i,:] = ["{:"+"{:d}".format(cs)+"s}" for cs in colums_width]
            else:
                formatting_string_array[i,:] = ["{:"+"{:d}".format(colums_width[0])+"s}"]+["{:>"+"{:d}".format(cs)+"s}" for cs in colums_width[1:]]
        
        # Format the string array
        for i in range(string_array.shape[0]):
            for j in range(string_array.shape[1]):
                string_array[i,j] = formatting_string_array[i,j].format(string_array[i,j])
        
        # Finally, add the walls of the displayed table, and create the final string
        string_header_line = "| {:s} |".format(" | ".join(string_array[0,:]))
        dash_line = "".join("-" for _ in range(len(string_header_line)))
        string_list = [dash_line, string_header_line, dash_line]
        string_list.extend("| {:s} |".format(" | ".join(string_array[i,:])) for i in range(1,string_array.shape[0]-2*int(agg_present)))
        if agg_present:
            string_list.append(dash_line)
            string_list.extend("| {:s} |".format(" | ".join(string_array[i,:])) for i in range(string_array.shape[0]-2,string_array.shape[0]))
        if l == 0:
            empty_line = "|{:s}|".format(("{:^"+str(len(string_header_line)-2)+"}").format("NaN"))
            string_list.append(empty_line)
        string_list.append(dash_line)
        final_string = "\n".join(string_list)
        
    return final_string

def _get_analogy_hyperparamaters_legend(analogy_type, K):
    hyperpameters_legend = "Analogy evaluation's hyperparameters:\n    analogy computation"\
    " type = '{:s}'\n    good if correct answer is within 'K' first fetched values: K = {:d}"
    hyperpameters_legend = hyperpameters_legend.format(analogy_type, K)
    return hyperpameters_legend

def _get_analogy_pre_report(analogy_evaluation_string, analogy_type, K):
    """
        Add information regarding hyperparameters and how to read the table, to an existing analogy evaluation string.
    """
    if analogy_evaluation_string != "":
        a_legend_string = AnalogyResult.get_format_legend()
        hyperpameters_legend = _get_analogy_hyperparamaters_legend(analogy_type, K)
        analogy_evaluation_string = "\n".join([hyperpameters_legend, " ", a_legend_string, analogy_evaluation_string])
    return analogy_evaluation_string

def _get_final_analogy_report(embeddings_legend, analogy_pre_report):
    """
        Create analogy evaluation report from embeddings legend string and analogy pre report.
    """
    strings_list = [ANALOGY_SECTION_TITLE, "Word embeddings' legend:", embeddings_legend, " ", analogy_pre_report]
    final_report = "\n".join(strings_list)
    return final_report

def format_analogy_results_table_multi_folder(analogy_results_lists_list_per_folder):
    analogy_string_list = []
    sorted_folder_names = sorted(analogy_results_lists_list_per_folder.keys())
    for folder_name in sorted_folder_names:
        analogy_results_lists_list = analogy_results_lists_list_per_folder[folder_name]
        analogy_evaluation_string =\
         format_analogy_results_table(analogy_results_lists_list, field_name=folder_name)
        analogy_string_list.append(analogy_evaluation_string)
    analogy_evaluation_string = "\n".join(analogy_string_list)
    
    return analogy_evaluation_string
    



def _additive_analogy(W, A, B, C):
    """
        :param W: the embedding's l2_normalized_matrix
        :param A: matrix whose columns correspond to the l2_normalized embedding of
         the first word of each analogy question
        :param B: matrix whose columns correspond to the l2_normalized embedding of
         the second word of each analogy question
        :param C: matrix whose columns correspond to the l2_normalized embedding of
         the third word of each analogy question
        :return y: a number_of_embedding's_words * number_of_asked_questions
         matrix, where each analogy question correspond to one column of y: for 
         each question (column), we want the value corresponding to the word that 
         is the answer to the analogy question to be the maximum value of the column.
        
        Here, for each question (a, b, c, d), we compute the l2 normalized vector 
        corresponding to b - a + c.
        Then we carry out the dot product of the result for each word in the 
        vocabulary of the embedding.
    """
    sum_matrix = B - A + C
    normalized_sum_matrix = l2_normalize(sum_matrix, rows=True)
    matrix_to_use = normalized_sum_matrix.T
    if (isinstance(W, numpy.ndarray) and isinstance(A, numpy.ndarray) and 
        isinstance(B, numpy.ndarray) and isinstance(C, numpy.ndarray)):
        y = np.dot(W, matrix_to_use)
    elif (isinstance(W, sparse.csr_matrix) and isinstance(A, sparse.csr_matrix) and 
          isinstance(B, sparse.csr_matrix) and isinstance(C, sparse.csr_matrix)):
        y = (W*matrix_to_use).A
    else:
        error_message = "All input matrix must be either numpy.ndarray, or "\
        "scipy.sparse.csr_matrix (here, input type = '{:s}')."
        raise ValueError(error_message)
        
    return y


def _cosmul_analogy(W, A, B, C):
    """
        :param W: the embedding's l2_normalized_matrix
        :param A: matrix whose columns correspond to the l2_normalized embedding of
         the first word of each analogy question
        :param B: matrix whose columns correspond to the l2_normalized embedding of
         the second word of each analogy question
        :param C: matrix whose columns correspond to the l2_normalized embedding of
         the third word of each analogy question
        :return y: a number_of_embedding's_words * number_of_asked_questions
         matrix, where each analogy question correspond to one column of y: for 
         each question (column), we want the value corresponding to the word that 
         is the answer to the analogy question to be the maximum value of the column.
        
        Here, for each question (a, b, c, d) and for each word w in the vocabulary 
        of the embedding, we compute cos(w, c) * cos(w, b) / (cos(w, a) + e), e small.
    """
    if (isinstance(W, numpy.ndarray) and isinstance(A, numpy.ndarray) and 
        isinstance(B, numpy.ndarray) and isinstance(C, numpy.ndarray)):
        y = numpy.divide(
                        numpy.multiply(
                                       (1+numpy.dot(W, C.T))/2,
                                       (1+numpy.dot(W, B.T))/2
                                       )
                        ,(1+numpy.dot(W, A.T))/2 +EPSILON_COSMUL
                        )
    elif (isinstance(W, sparse.csr_matrix) and isinstance(A, sparse.csr_matrix) and 
          isinstance(B, sparse.csr_matrix) and isinstance(C, sparse.csr_matrix)):
        y = numpy.divide(
                        numpy.multiply(
                                       (1+(W*C.T).A)/2,
                                       (1+(W*B.T).A)/2
                                       )
                        ,(1+(W*A.T).A)/2 +EPSILON_COSMUL
                        )
    else:
        error_message = "All input matrix must be either numpy.ndarray, or."
        raise ValueError(error_message)
    
    return y


def _compute_analogy_one_file(embedding, filepath, analogy_type="cosmul", K=1, drop_raw_data=True):
    # compute_results_stats=False: We give ourselves the possibility to use the results in a finer way than just already comuting the stats and discarding the data
    """
        Evaluates embedding on analogies dataset (a - b + c = d)
        :param embeddings:
        :param filepath: path to a text file containing an analogy questions dataset
        :param analogy_type: specify which method ('cosmul' or 'additive') to use 
        to compute value used to find the answer to an analogy question 
        (cosmul by default)
        :return: a dictionary defining the name of the file containing the dataset,
         as well as a list of the possible questions contained in the dataset, a 
         list of the questions actually asked, a list of the correctly answered 
         ones, and a list of the incorrectly answered one.
    """
    if analogy_type not in POSSIBLE_ANALOGY_TYPE:
        error_message = "Incorrect 'analogy_type' input; possible inputs are:{:s}".format(str(POSSIBLE_ANALOGY_TYPE))
        raise ValueError(error_message)
    
    word_index = embedding.vocabulary.word_index
    
    a, b, c, d = [], [], [], []  # lists containing the index of the terms of the question
    possible_questions = []
    file_vocabulary = set()
    file_vocab_size = 0
    embedding_common_vocab_size = numpy.NaN
    asked_questions = []
    correct = []
    incorrect = []
    line_nb = -1
    temp_logging_level = -10
    
    # filter lines where all words appear in our target words
    with InputOutputLoggerTimer(logger, temp_logging_level, None, 
                                "Reading file '{:s}' for embedding 'label = {:s}' took {:s}.".format(os.path.basename(filepath), embedding.label, "{:s}")):
        try:
            with open(filepath, "r") as f:
                for line_nb, line in enumerate(f):
                    word_list = line.strip().split()
                    try:
                        word1, word2, word3, word4 = word_list
                    except Exception as e:
                        error_message =\
                         "Incorrect analogy question (line_nb = {:d}, line = '{:s}')".format(line_nb, line)
                        raise ValueError(error_message)
                    possible_questions.append(word_list)
                    [file_vocabulary.add(w) for w in word_list]
                    if all(w in word_index for w in word_list):
                        a.append(word_index[word1])
                        b.append(word_index[word2])
                        c.append(word_index[word3])
                        d.append(word_index[word4])
                        asked_questions.append(word_list)
        except OSError as e:
            raise e
    
    with InputOutputLoggerTimer(logger, temp_logging_level, 
                                "Answering for embedding's label = {:s}' on file '{:s}' took {:s}".format(embedding.label, os.path.basename(filepath), "{:s}")):
        if line_nb == -1:
            a_message =\
             "The file located at '{:s}' is empty, cannot ask analogy questions.".format(filepath)
            logger.warning(a_message)
        else:
            # Number of questions
            number_of_questions = len(a)
            file_vocab_size = len(file_vocabulary)
            embedding_common_vocab_size = len(file_vocabulary.intersection(set(word_index.keys())))
    
            if number_of_questions == 0:
                a_message = "Vocabulary not rich enough for embedding (label "\
                "= {:s}), cannot ask a single question among the possible ones in '{:s}'."
                a_message = a_message.format(embedding.label, filepath)
                logger.debug(a_message)
            else:
                if not embedding.is_l2_normalized():
                    embedding.l2_normalize()
            
                if analogy_type == "cosmul":
                    analogy_compute_method = _cosmul_analogy
                    used_matrix = embedding.l2_normalized_matrix
                elif analogy_type == "additive":
                    analogy_compute_method = _additive_analogy
                    used_matrix = embedding.l2_normalized_matrix
                    
                A = used_matrix[a,:]
                B = used_matrix[b,:]
                C = used_matrix[c,:]
                W = used_matrix
                
                y = analogy_compute_method(W, A, B, C)
                
                nb_words = W.shape[0]
                for j in range(number_of_questions):
                    # find the best response of each question according to our embeddings
                    # i.e. find the max value in current column, excluding the 3 first words of the question
                    best_answers = nlargest(K, [i for i in range(nb_words) if i not in (a[j], b[j], c[j])], key=lambda x: y[x, j])
                    word_list = asked_questions[j]
                    if d[j] in best_answers:
                        # if the found response is the one expected, accuracy ++
                        correct.append(word_list)
                    else:
                        index_best = best_answers[0]
                        if (index_best in embedding.vocabulary.index_word):
                            best_word = embedding.vocabulary.index_word[index_best]
                        else:
                            best_word = "OOV word"
                        incorrect_word_list = word_list[:3]+[best_word]
                        incorrect.append(incorrect_word_list)
            
    file_name = os.path.basename(filepath)
    total_questions_nb = len(possible_questions)
    a_result = AnalogyResult(file_name, total_questions_nb, file_vocab_size,
                             embedding_common_vocab_size, correct, incorrect,
                             drop_raw_data=drop_raw_data)
    
    return a_result

def _analogy_file_evaluation(embeddings_collection, analogy_dataset_file_path,
                              analogy_type="cosmul", K=1, drop_raw_data=True):
    """
        Carry out analogy evaluation on the file located at the input file path.
        Returns:
            - a string containing the table of results to be displayed
            - the corresponding embeddings' legend
            - the list of the respective embeddings' results / performance on this file
    """
    with InputOutputLoggerTimer(logger, "info", 
                                "evaluating analogy with data from the file located at'{:s}'".format(analogy_dataset_file_path)): 
        # Compute the analogy evaluation result(s)
        analogy_results_lists_list = []
        for embedding in embeddings_collection:
            an_analogy_result =\
             _compute_analogy_one_file(embedding, analogy_dataset_file_path,
                                       analogy_type=analogy_type,
                                       K=K, drop_raw_data=drop_raw_data)
            analogy_results_lists_list.append([an_analogy_result])
        
        # Format the analogy evaluation pre report
        analogy_evaluation_string = format_analogy_results_table(analogy_results_lists_list, field_name="")
        # Get the embeddings legend string, at the same spot where the analogy evaluation string is generated
        legend_string = AnalogyResult._get_legend_string(embeddings_collection)
    
    return analogy_evaluation_string, legend_string, an_analogy_result

def analogy_file_evaluation(embeddings_collection, analogy_dataset_file_path,
                              analogy_type="cosmul", K=1, drop_raw_data=True,
                              output_results=False):
    """
        Carry out analogy evaluation on the file located at the input file path.
        Return a report detailing the analogy performance of each embedding on this file.
    """
    analogy_evaluation_string, legend_string, an_analogy_result =\
     _analogy_file_evaluation(embeddings_collection, analogy_dataset_file_path,
                              analogy_type=analogy_type, K=K,
                              drop_raw_data=drop_raw_data)
    
    analogy_pre_report = _get_analogy_pre_report(analogy_evaluation_string, analogy_type, K)
    analogy_report = _get_final_analogy_report(legend_string, analogy_pre_report)
    
    if output_results:
        return analogy_report, an_analogy_result
    else:
        return analogy_report


def _analogy_one_folder_evaluation(embeddings_collection, analogy_dataset_folder_path,
                              analogy_type="cosmul", K=1, drop_raw_data=True,
                              workers_nb=1, temp_folder_path=None):
    """
        Carry out analogy evaluation on all files contained within the folder located at the input folder path.
        Returns:
            - a string containing the table of results to be displayed (including  aggregated  stats 'micro-average' and 'macro-average')
            - the corresponding embeddings' legend
            - the list of the respective embeddings' results / performance lists on this folder.
    """
    with InputOutputLoggerTimer(logger, "info", 
                                "evaluating analogy with data from the folder located at '{:s}'".format(analogy_dataset_folder_path)):
        
        dataset_folder_path = analogy_dataset_folder_path
        analogy_domain = get_folder_name(dataset_folder_path)
        
        # Get results
        analogy_results_lists_list_dict =\
         _compute_evaluation_folders_multiproc(embeddings_collection,
            analogy_dataset_folder_path, _compute_analogy_one_file,
            analogy_type=analogy_type, K=K, drop_raw_data=drop_raw_data,
            workers_nb=workers_nb, subfolder=False,
            temp_folder_path=temp_folder_path)
        analogy_results_lists_list = list(analogy_results_lists_list_dict.values())[0]
        
        # Format results
        analogy_evaluation_string = format_analogy_results_table(analogy_results_lists_list, field_name=analogy_domain)
        # Get the embeddings legend string, at the same spot where the analogy evaluation string is generated
        legend_string = AnalogyResult._get_legend_string(embeddings_collection)
        
    return analogy_evaluation_string, legend_string, analogy_results_lists_list

def analogy_one_folder_evaluation(embeddings_collection, analogy_dataset_folder_path,
                              analogy_type="cosmul", K=1, drop_raw_data=True,
                              workers_nb=1, temp_folder_path=None,
                              output_results=False):
    """
        Carry out analogy evaluation on all files contained within the folder located at the input folder path.
        Return a report detailing the performance on each files, as well as aggregated 
        stats 'micro-average' and 'macro-average' for this folder.
    """
    analogy_evaluation_string, legend_string, analogy_results_lists_list =\
     _analogy_one_folder_evaluation(embeddings_collection, analogy_dataset_folder_path,
                analogy_type=analogy_type, K=K, drop_raw_data=drop_raw_data,
                workers_nb=workers_nb, temp_folder_path=temp_folder_path)
    
    analogy_pre_report = _get_analogy_pre_report(analogy_evaluation_string, analogy_type, K)
    analogy_report = _get_final_analogy_report(legend_string, analogy_pre_report)
    
    if output_results:
        return analogy_report, analogy_results_lists_list
    else:
        return analogy_report


def _analogy_evaluation(embeddings_collection, language="en",
                        analogy_dataset_folder_path="google",
                        analogy_type="cosmul", K=1, drop_raw_data=True,
                        workers_nb=1, temp_folder_path=None):
    """
        Carry out analogy evaluation on all subfolder contained within the folder located at the input folder path.
        Returns:
            - a string containing the table of results to be displayed (including aggregated  stats 'micro-average' and 'macro-average' for each subfolder)
            - the corresponding embeddings' legend
            - the dict of each subfolder's list of the respective embeddings' results / performance lists.
    """
    # Get correct dataset folder path
    if analogy_dataset_folder_path in ANALOGY_OWN_ENGLISH_DATASETS:
        dataset_name = analogy_dataset_folder_path
        datasets_folder_path = _check_language_folder(language)
        analogy_dataset_folder_path = os.path.join(datasets_folder_path, "analogy")
        if language == "en":
            analogy_dataset_folder_path = os.path.join(analogy_dataset_folder_path,
                                                       dataset_name)
    
    # Get results
    with InputOutputLoggerTimer(logger, "info", 
                                "carrying out analogy evaluation for the embeddings collection ({:d} embeddings)".format(len(embeddings_collection))):
        analogy_results_lists_list_dict =\
             _compute_evaluation_folders_multiproc(embeddings_collection,
                analogy_dataset_folder_path, _compute_analogy_one_file,
                analogy_type=analogy_type, K=K, drop_raw_data=drop_raw_data,
                workers_nb=workers_nb, subfolder=True,
                temp_folder_path=temp_folder_path)
    
    # Format results
    analogy_evaluation_string = format_analogy_results_table_multi_folder(analogy_results_lists_list_dict)
    # Get the embeddings legend string, at the same spot where the analogy evaluation string is generated
    legend_string = AnalogyResult._get_legend_string(embeddings_collection)
    
    return analogy_evaluation_string, legend_string, analogy_results_lists_list_dict


def analogy_evaluation(embeddings_collection, language="en", analogy_dataset_folder_path="google",
                       analogy_type="cosmul", K=1, drop_raw_data=True,
                       workers_nb=1, temp_folder_path=None,
                       output_results=False):
    """
        Computes the results of the analogy performance tasks for each item of 
        an embedding object collection.
        :param embeddings_collection: a liste of Embedding objects
        :param analogy_dataset_folder_path: the path to the folder containing 
        the subfolders and / or files to use in order to perform the computation.
        :param analogy_type: "cosmul" or "additive"
        :param K: Minimal rank of the embedding's answer to an analogy question
         in order to be considered a correct answer.
        :param workers_nb: Number of processes to use (allows multiprocessing).
        :return: (analogy_pre_report, legend_string, analogy_results_lists_dict) 
        tuple: the analogy_pre_report is an easy-to-read string representations 
        of the results, and the legend_string is the legend string.
    """    
    analogy_evaluation_string, legend_string, analogy_results_lists_dict =\
     _analogy_evaluation(embeddings_collection, language, analogy_dataset_folder_path,
                         analogy_type=analogy_type, K=K,
                         drop_raw_data=drop_raw_data, workers_nb=workers_nb,
                         temp_folder_path=temp_folder_path)
    
    analogy_pre_report = _get_analogy_pre_report(analogy_evaluation_string, analogy_type, K)
    analogy_report = _get_final_analogy_report(legend_string, analogy_pre_report)
    
    if output_results:
        return analogy_report, analogy_results_lists_dict
    else:
        return analogy_report






def evaluate(embeddings_collection, language="en", max_vocab_size=-1,
             similarity_dataset_folder_path=None,
             analogy_dataset_folder_path="google", analogy_type="cosmul", K=1,
             drop_raw_data=True, workers_nb=1, temp_folder_path=None,
             output_results=False):
    """
        Use this function to launch every evaluation process possible on all the input embeddings, and return an agreggated report.
        => One needs to input as much parameters as the union of all those needed by each evaluation process possible.
        
    """
    if max_vocab_size > 0:
        max_vocab_size = int(max_vocab_size)
        for i, embedding in enumerate(embeddings_collection):
            embedding = embedding.get_pruned_copy(max_vocab_size)
            embedding.label = "{} - truncated to its {:d} first words".format(embedding.label, max_vocab_size)
            embeddings_collection[i] = embedding
        
    similarity_evaluation_string, legend_string, similarity_results_lists_dict =\
     _similarity_evaluation(embeddings_collection, language,
                           similarity_dataset_folder_path,
                           drop_raw_data=drop_raw_data,
                           workers_nb=workers_nb,
                           temp_folder_path=temp_folder_path)
    similarity_pre_report = _get_similarity_pre_report(similarity_evaluation_string)
    
    analogy_evaluation_string, _, analogy_results_lists_dict =\
     _analogy_evaluation(embeddings_collection, language,
                        analogy_dataset_folder_path,
                        analogy_type=analogy_type, K=K, drop_raw_data=drop_raw_data,
                        workers_nb=workers_nb,
                        temp_folder_path=temp_folder_path)
    analogy_pre_report = _get_analogy_pre_report(analogy_evaluation_string, analogy_type, K)
    
    # Evaluation report
    strings_list = [REPORT_TITLE, "Word embeddings' legend:", legend_string, " ", ANALOGY_SECTION_TITLE, analogy_pre_report,
                    " ", SIMILARITY_REPORT_TITLE, similarity_pre_report]
    final_string = "\n".join(strings_list)
    
    if output_results:
        # Dict of original results
        evaluation_dict = {"analogy": analogy_results_lists_dict, "similarity": similarity_results_lists_dict}
        return final_string, evaluation_dict
    else:
        return final_string


def _check_language_folder(language):
    filename_list = list_subdirectories(EVALUATION_FOLDER_PATH)
    if not (language in POSSIBLE_LANGUAGE_LIST):
        error_message = "Language not implemented (as far the evaluation is "\
        "concerned). Implemented languages are: {:s}."
        error_message = error_message.format(",".join(POSSIBLE_LANGUAGE_LIST))
        raise ValueError(error_message)
    corresponding_folder_name = language.lower()
    if not (corresponding_folder_name in filename_list):
        error_message = "Problem: the evaluation data folder corresponding to"\
         " this language is not present."
        raise ValueError(error_message)
    
    datasets_folder_path =\
     os.path.join(EVALUATION_FOLDER_PATH, corresponding_folder_name)
    
    return datasets_folder_path



def evaluate_a_word_embeddings(embedding, max_vocab_size=-1, language="en",
                               similarity_dataset_folder_path=None,
                               analogy_dataset_folder_path="google", analogy_type="cosmul", K=1,
                               drop_raw_data=True,
                               workers_nb=1,
                               seed=42,
                               temp_folder_path=None,
                               output_results=True):
    used_embedding = embedding
    if max_vocab_size > 0:
        max_vocab_size = int(max_vocab_size)
        used_embedding = embedding.get_pruned_copy(max_vocab_size)
        used_embedding.label = "{} - truncated to its {:d} first words".format(embedding.label, max_vocab_size)
    
    embeddings_collection = [used_embedding]
    final_string, evaluation_dict = evaluate(embeddings_collection, language=language, 
                                             max_vocab_size=max_vocab_size, 
                                             similarity_dataset_folder_path=similarity_dataset_folder_path,
                                             analogy_dataset_folder_path=analogy_dataset_folder_path, 
                                             analogy_type=analogy_type, 
                                             K=K,
                                             drop_raw_data=drop_raw_data, 
                                             workers_nb=workers_nb,
                                             temp_folder_path=temp_folder_path,
                                             output_results=True)
    
    if output_results:
        return final_string, evaluation_dict
    else:
        return final_string



def main():
    """The entry point for script_evaluation.py
    
    This function parses the command line arguments using an instance of ArgumentParser, and then 
    carry out its task: to load the embedding and evaluate it.
    :return: 0 if everything is ok
    """        
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    
    parser.add_argument("-v", help="Display 'info' level logs on console", action="store_true", 
                        dest="verbose")
    parser.add_argument("-vv", help="Display 'debug' level logs on console", action="store_true", 
                        dest="superverbose")
    parser.add_argument("-q", "--quiet", help="Display no logs on console", action="store_true", dest="quiet")
    parser.add_argument("-m", type=int, default=1, dest="threads_nb", 
                        help="Number of subprocess(es) to use (1 by default - no multiprocessing)")
    
    
    parser.add_argument("-f", "--embedding_folders", dest="embedding_folders", nargs="+", 
                        help="Path to folder containing the files of the embedding to evaluate "\
                             "(folder containing a vocabulary.txt file and a matrix.npy file)")
    parser.add_argument("-t", "--embedding_text_files", dest="embedding_text_files", nargs="+", 
                        help="Path to the embedding as a (possibly gzip-compressed) text file")
    
    
    parser.add_argument("-r", "--eval_report_dir", default=os.path.dirname(os.path.abspath(__name__)), 
                        dest="eval_reports_dir",
                        help="Path to evaluation reports folder, where reports are to be written (in the same folder as this script by default))")
    """
    parser.add_argument("--label", dest="label", default=DEFAULT_LABEL,
                        help="Label used to identify the embedding in  the evaluation report")
    """
    
    parser.add_argument("-l", "--language", dest="language", choices=POSSIBLE_LANGUAGE_LIST, 
                        default="en",
                        help="Language of the word embedding (as ISO 639-1 string, such as 'en' or 'fr'; 'en' by default)")
    
    parser.add_argument("-s", "--max_vocab_size", type=int, default=-1, dest="max_vocab_size", 
                        help="Length of vocabulary to truncate the embedding with for the duration of the evaluation - faster evaluation but less question coverage. None by default.")
    parser.add_argument("-k", type=int, default=1, dest="K", 
                        help="Value of parameter 'K' in analogy evaluation (positive integer, 1 by default)")
    parser.add_argument("-a", "--analogy_type", dest="analogy_type", choices=POSSIBLE_ANALOGY_TYPE, 
                        default="cosmul", help="Type of analogy computation ('cosmul' by default)")
    parser.add_argument("--english_analogy_dataset", dest="english_analogy_dataset", 
                        choices=ANALOGY_OWN_ENGLISH_DATASETS, 
                        default="google", help="Analogy evaluation data set for English language ('google' by default)")
    
    # Parsing of the arguments
    args = parser.parse_args()
    
    # If the user asked for verbose output
    logging_level = logging.WARNING
    if args.verbose:
        logging_level = logging.INFO
    if args.superverbose:
        logging_level = logging.DEBUG
    if args.quiet:
        logging_level = logging.FATAL + 1
    
    # Logging configuration
    logging.basicConfig(format='%(asctime)s :: %(levelname)-8s ::  %(message)s', 
                        level=logging_level, handlers=[logging.StreamHandler()]) 
    
    # Get the word embeddings language value
    max_vocab_size = args.max_vocab_size
    language = args.language
    analogy_type = args.analogy_type
    K = max(1, args.K)
    workers_nb = max(1, args.threads_nb)
    english_analogy_dataset = args.english_analogy_dataset
    
    # Check if the output directory exists
    if os.path.exists(args.eval_reports_dir):
        eval_reports_dir = args.eval_reports_dir
    else:
        print("The evaluation reports folder '{}' doesn't exist.".format(args.eval_reports_dir))
        return 1
    
    # Check existence of embedding, load it if it does exist
    embedding_folders = args.embedding_folders
    embedding_text_files = args.embedding_text_files
    embeddings_collection = []
    if embedding_folders is not None:
        for folder_path in embedding_folders:
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                label = get_folder_name(folder_path)
                embedding = Embedding.load_from_folder(folder_path, label=label)
                embeddings_collection.append(embedding)
                # TODO : test if files exist
            else:
                print("The embedding folder '{}' doesn't exist / is not a folder.".format(folder_path))
                return 2
    
    if embedding_text_files is not None:
        for file_path in embedding_text_files:
            if os.path.exists(file_path) and os.path.isfile(file_path):
                label = os.path.basename(file_path)
                embedding = Embedding.load_from_text_file(file_path, label=label)
                embeddings_collection.append(embedding)
            else:
                print("The embedding file '{}' does not exist / is not a file.".format(file_path))
                return 2
    
    if len(embeddings_collection) < 1:
        print("No embedding were specified / embedding list is empty.")
        return 3
    
    
    # Time to evaluate the hell outta it!
    """
    report_string = evaluate_a_word_embeddings(embedding, language=language,
                                              max_vocab_size=max_vocab_size,
                                              similarity_dataset_folder_path=None,
                                              analogy_dataset_folder_path=english_analogy_dataset,
                                              analogy_type=analogy_type, K=K,
                                              drop_raw_data=True,
                                              workers_nb=workers_nb,
                                              output_results=False)
    """
    report_string = evaluate(embeddings_collection, language=language, 
                             max_vocab_size=max_vocab_size,
                             similarity_dataset_folder_path=None,
                             analogy_dataset_folder_path=english_analogy_dataset, 
                             analogy_type=analogy_type, K=K,
                             drop_raw_data=True, 
                             workers_nb=workers_nb, 
                             temp_folder_path=None,
                             output_results=False)
    
    # Such a magnificent evaluation report to write!
    report_filepath = os.path.join(eval_reports_dir, REPORT_FILENAME)
    with open(report_filepath, "wb") as f:
        f.write(report_string.encode("utf-8"))
        print("Report written in '{:s}'.".format(report_filepath))
    
    print(report_string)

    return 0


if __name__ == '__main__':
    
    main()
