import sys
import subprocess
import re
import dask
import numpy as np
import xarray as xr
import zarr as zr
import os.path
import shutil
import glob
import ntpath
import datetime
import os
import uuid
import json

from dask.distributed import Client

from matplotlib import pyplot as plt, colors
from matplotlib.colors import LinearSegmentedColormap, Colormap
import math
from numcodecs import Blosc

input_folder = "\in_data"
output_folder = "\out_data"

simrad_color_table = [(1, 1, 1),
                                        (0.6235, 0.6235, 0.6235),
                                        (0.3725, 0.3725, 0.3725),
                                        (0, 0, 1),
                                        (0, 0, 0.5),
                                        (0, 0.7490, 0),
                                        (0, 0.5, 0),
                                        (1, 1, 0),
                                        (1, 0.5, 0),
                                        (1, 0, 0.7490),
                                        (1, 0, 0),
                                        (0.6509, 0.3255, 0.2353),
                                        (0.4705, 0.2353, 0.1568)]
simrad_cmap = (LinearSegmentedColormap.from_list
                                ('Simrad', simrad_color_table))
simrad_cmap.set_bad(color='grey')

def plot_da(input, out_name, range_res = 600, time_res = 800, interpolate = False):
        # Prepare simrad cmap
    range_len = len(input.range)
    time_len = len(input.ping_time)

    if range_len > range_res or time_len > time_res:
        mult_range = math.floor(range_len/range_res)
        mult_time = math.floor(time_len/time_res)

        if mult_range == 0:
            mult_range = 1

        if mult_time == 0:
            mult_time = 1

        if interpolate == False:
            input = input[:, ::mult_time,::mult_range]
        else:
            input = input.coarsen(range = mult_range, ping_time = mult_time, boundary="trim").mean(skipna=True)

    vmin = input.min(skipna=True).compute()
    vmax = input.max(skipna=True).compute()

    # Handle duplicate frequencies
    if len(input.frequency.data) == len(np.unique(input.frequency.data)):
        input.plot(x="ping_time", y="range", row= "frequency", vmin = vmin, vmax = vmax, norm=colors.LogNorm(), cmap=simrad_cmap)
    else:
        frstr = ["%.2f" % i for i in input.frequency.data]
        new_coords = []
        for frname in frstr:
            orig = frname
            i = 1
            while frname in new_coords:
                frname = orig + " #" + str(i)
                i += 1
            new_coords.append(frname)
        input.coords["frequency"] = new_coords
        input.plot(x="ping_time", y="range", row= "frequency", vmin = vmin, vmax = vmax, norm=colors.LogNorm(), cmap=simrad_cmap)

    plt.gca().invert_yaxis()
    plt.gcf().set_size_inches(8,11)
    plt.savefig(out_name + "." + 'png', bbox_inches = 'tight', pad_inches = 0)

def plot_da_2D(input, out_name, range_res = 600, time_res = 800, interpolate = False, incmap = simrad_cmap):
    range_len = len(input.range)
    time_len = len(input.ping_time)

    if range_len > range_res or time_len > time_res:
        mult_range = math.floor(range_len/range_res)
        mult_time = math.floor(time_len/time_res)

        if mult_range == 0:
            mult_range = 1
            print("mult range")

        if mult_time == 0:
            mult_time = 1
            print("no mult range")

        if interpolate == False:
            input = input[::mult_time,::mult_range]
        else:
            input = input.coarsen(range = mult_range, ping_time = mult_time, boundary="trim").mean(skipna=True)

    vmin = input.min(skipna=True).compute()
    vmax = input.max(skipna=True).compute()
    input.plot(x='ping_time', cmap = incmap)
    plt.gca().invert_yaxis()
    plt.gcf().set_size_inches(8,11)
    plt.savefig(out_name + "." + 'png', bbox_inches = 'tight', pad_inches = 0)

def plot_da_2D_Bool(input, out_name, range_res = 600, time_res = 800, interpolate = False):
    range_len = len(input.range)
    time_len = len(input.ping_time)

    if range_len > range_res or time_len > time_res:
        mult_range = math.floor(range_len/range_res)
        mult_time = math.floor(time_len/time_res)

        if mult_range == 0:
            mult_range = 1
            print("mult range")

        if mult_time == 0:
            mult_time = 1
            print("no mult range")

        if interpolate == False:
            input = input[::mult_time,::mult_range]
        else:
            input = input.coarsen(range = mult_range, ping_time = mult_time, boundary="trim").mean(skipna=True)

    vmin = input.min(skipna=True).compute()
    vmax = input.max(skipna=True).compute()
    
    input.plot(x='ping_time', cmap='Greys', label='_nolegend_')
    plt.gcf().set_size_inches(8,11)
    plt.gca().invert_yaxis()
    plt.axis("tight") 
    plt.tight_layout()
    plt.savefig(out_name + "." + 'png', bbox_inches = 'tight', pad_inches = 0)
def plot_all(ds, out_name, range_res = 600, time_res = 800, interpolate = False):
    # Prepare simrad cmap
    sv = ds.sv
    plot_da(sv, out_name, range_res, time_res, interpolate)

def classify_zarr(upper, lower, in_name, out_name, input_folder, output_folder):
    # Check whether the specified path exists or not
    isExist = os.path.exists(output_folder)

    if not isExist:
  
    # Create a new directory because it does not exist 
        os.makedirs(output_folder)
        print("The new directory is created!")

    fullpath = os.path.join(input_folder, in_name)
    ds = xr.open_zarr(fullpath, chunks={'ping_time':'auto'}, consolidated=False)
    sv = ds.sv
    # plot_all(ds, 'out_name_raw')

    sv38 = sv[0,:,:]
    # print(sv38)
    sv200 = sv[1,:,:]

    plot_da_2D(sv38, os.path.join(output_folder, out_name) + "_38")

    # vmin = sv38.min(skipna=True).compute()
    # vmax = sv38.max(skipna=True).compute()
    # print("Min 38:" + str(vmin))
    # print("Max 38:" + str(vmax))

    svdiff = sv38 - sv200
    # vmin = svdiff.min(skipna=True).compute()
    # vmax = svdiff.max(skipna=True).compute()
    # # print("Min Diff:" + str(vmin))
    # # print("Max diff:" + str(vmax))
    plot_da_2D(svdiff, os.path.join(output_folder, out_name) + "_diff")

    # vmin = svdiff.min(skipna=True).compute()
    # vmax = svdiff.max(skipna=True).compute()
    # print("Min Diff:" + str(vmin))
    # print("Max diff:" + str(vmax))
    
    
    fishDetect = (svdiff > lower) & (svdiff < upper)
    # print(fishDetect)
    # vmin = fishDetect.min(skipna=True).compute()
    # vmax = fishDetect.max(skipna=True).compute()
    # print("Min Fish:" + str(vmin))
    # print("Max Fish:" + str(vmax))
    
    # fish = svdiff.sel(sv = slice(0,2)) 
    plot_da_2D_Bool(fishDetect, os.path.join(output_folder, out_name) + "_fish")


    predDs = xr.Dataset({'acousticFishDetections': fishDetect})
    #predDs = xr.Dataset()

    # print(predDs)

    #predDs = fish.to_dataset()

    #predDs.assign(fish)
    newfilename = os.path.join(output_folder, out_name + ".zarr")
    if os.path.exists(newfilename):
        newfilename = os.path.join(output_folder, str(uuid.uuid4()) + out_name + ".zarr")

    predDs.to_zarr(newfilename)

    
#load configuration
with open('config.json', 'r') as f:
    config = json.load(f)


upperThreshold = config['upperThreshold']
lowerThreshold = config['lowerThreshold']
# scan for zarr files in folder.
directory = os.fsencode(input_folder)
    
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".zarr"): 
         print("processing file " + filename + " with " + str(upperThreshold) + " to " + str(lowerThreshold))
         classify_zarr(upperThreshold, lowerThreshold, filename, filename + "_pred", input_folder, output_folder)
         continue
     else:
         continue

