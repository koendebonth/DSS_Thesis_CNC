#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""





"""

""" 
# Copyright (c) 2021 Bosch Rexroth AG
# SPDX-License-Identifier: BSD-3-Clause

Author: Mohamed-Ali Tnani (Mohamed-ali.tnani@boschrexroth.de)
Date: November 24, 2021

This software is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the copyright holder or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
import datetime
import logging

# Configuration for logging
def setup_logging():
    """
    Configure logging to write to a file instead of the console.
    Creates a 'logs' directory in the root of the project if it doesn't exist yet.
    """
    # Get the project root directory by going up two levels from utils/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"data_loader_{datetime.datetime.now().strftime('%d-%m-%y')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
        ]
    )
    
    return logging.getLogger('data_loader_utils')

# Initialize logger
logger = setup_logging()

def find_all_h5s_in_dir(s_dir: os.PathLike):
    """
    list all .h5 files in a directory
    """

    fileslist = []
    for root, dirs, files in os.walk(s_dir):
        for file in files:
            if file.endswith(".h5"):
                fileslist.append(file)
    return fileslist


def load_tool_research_data(data_path: os.PathLike, label: Literal["good","bad"], add_additional_label: bool =True, verbose: bool =True):
    """
    load data (good and bad) from the research data storages
    
    Keyword Arguments:
            data_path {str} -- [path to the directory] 
            label {str} -- ["good" or "bad"]
            add_additional_label {bool} -- [if true the labels will be in the form of "Mxx_Aug20xx_OPxx_sampleNr_label" otherwise "label"] (default: True)
            verbose {bool}

        Returns:
            datalist --  [list of the the X samples]
            label --  [list of the the y labels ]
    """
    datalist = []
    data_label = []

    # list all .h5 files
    list_paths = find_all_h5s_in_dir(data_path)
    list_paths.sort()
    if not list_paths and verbose:
        logger.warning(f"skipping {data_path} empty directory...")

    # read and append the samples with the corresponding labels
    if verbose:
        logger.info(f"loading files from {data_path}... ")
    for element in list_paths:
        # check if additional label needed ("Mxx_Aug20xx_Tool,nrX") 
        if add_additional_label:
            add_label = element.split('/')[-1]
            additional_label = add_label[:-3] + "_" + label
        else:
            additional_label = label
        # extract data X and y 
        with h5py.File(os.path.join(data_path, element), 'r') as f:
            vibration_data = f['vibration_data'][:]
        datalist.append(vibration_data)
        data_label.append(additional_label)

    return datalist, data_label


def datafile_read(file: os.PathLike, plotting=True, axes=None):
    """loads and plots the data from the datafile

    Keyword Arguments:
        file {str} -- [path of the file]
        plotting {bool} -- [whether to plot the data]
        axes {list} -- [list of axes to plot, e.g. [0,2] for X and Z axes. If None, plots all axes]

    Returns:
        ndarray --  [raw data original]
    """
    with h5py.File(file, 'r') as f:
        vibration_data = f['vibration_data'][:]
    # interpolation for x axis plot
    freq = 2000
    samples_s = len(vibration_data[:, 0]) / freq
    samples = np.linspace(0, samples_s, len(vibration_data[:, 0]))

    # plotting
    if plotting:
        if axes is None:
            axes = [0, 1, 2]  # Plot all axes by default
            
        axis_names = ['X', 'Y', 'Z']
        for axis in axes:
            plt.figure(figsize=(20, 5))
            plt.plot(samples, vibration_data[:, axis], 'b')
            plt.ylabel(f'{axis_names[axis]}-axis Vibration Data')
            plt.xlabel('Time [sec]')
            plt.locator_params(axis='y', nbins=10)
            plt.grid()
            plt.show()
    return vibration_data
