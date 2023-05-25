# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 15:30:40 2022

@author: T pro
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 15:09:08 2022

@author: T pro
"""
import msgspec
from msgspec.json import decode
from msgspec import Struct
from typing import Set
import os
import json
import time
import numpy as np
from joblib import Parallel, delayed
import sys
#--------------- N E T W O R K ---------------------------
print("----------------TRY----------------")
class Network(Struct):
    network: dict
    
def net(dictionary, feature):
    try:
        with open(dictionary, "r") as f:
             data = f.read()
             byte_data = (bytes(data, 'utf-8'))
             jsn = msgspec.json.decode(byte_data, type=Network)
             for key in jsn.network:
                 if key == feature:
                     return len(jsn.network[feature])
    except:
        return 0

network_names = ["icmp", "tcp", "udp", "http"]

     
def network_calls(dictionary):
    net_vector = []
    for name in network_names:
        net_vector.append(net(dictionary, name))
    return net_vector
        
#network_full = network_calls(r"C:/Users/T pro/Desktop/ffe7f8921b0d42d3014c972c33460ec0f6565e4ce0bb8bd04df4a6d635c7903d.json")

#--------------------------S T A T I C -------------------------
class PE(Struct):
    pe: dict
class Static(Struct):
    static: PE

def stat_pe(dictionary, feature):
    with open(dictionary, "r") as f:
         data = f.read()
         byte_data = (bytes(data, 'utf-8'))
         try: 
             print('yes')
             jsn = msgspec.json.decode(byte_data, type=Static)
             for key in jsn.static:
                 if key == feature:
                     return len(jsn.static[feature])
         except:
            print('no')
            return 0

static_names = ["pe_sections", "pe_exports", "pe_resources"]

def static_calls(dictionary):
    static_vector = []
    for name in static_names:
        static_vector.append(stat_pe(dictionary, name))
    return static_vector

#static_full = static_calls(r"C:/Users/T pro/Desktop/ffe7f8921b0d42d3014c972c33460ec0f6565e4ce0bb8bd04df4a6d635c7903d.json")


#------------------------------ B E H A V I O U R -----------------
class Behavior(Struct):
    behavior: dict
def behav_calls(dictionary):
    with open(dictionary, "r") as f:
         data = f.read()
         byte_data = (bytes(data, 'utf-8'))
         try: 
             jsn = msgspec.json.decode(byte_data, type=Behavior)
             return len(jsn.behavior["processes"])
         except:
            return 0

#behavior_full = behav_calls(r"C:/Users/T pro/Desktop/ffe7f8921b0d42d3014c972c33460ec0f6565e4ce0bb8bd04df4a6d635c7903d.json")

#------------------------- D R O P P E D -------------------------
class Dropped(Struct):
    dropped: dict
def dropped_calls(dictionary):
    with open(dictionary, "r") as f:
         data = f.read()
         byte_data = (bytes(data, 'utf-8'))
         try: 
             jsn = msgspec.json.decode(byte_data, type=Dropped)
             return len(jsn.behavior)
         except:
            return 0

#dropped_full = dropped_calls(r"C:/Users/T pro/Desktop/ffe7f8921b0d42d3014c972c33460ec0f6565e4ce0bb8bd04df4a6d635c7903d.json")

#-------------------------- R E G I S T R Y  K E Y S ----------------------

class Enhanced(Struct):
    enhanced: list
class BehaviorReg(Struct):
    behavior: Enhanced
    
def registry_finder(dictionary, feature):
    counter = 0
    with open(dictionary, "r") as f:
         data = f.read()
         byte_data = (bytes(data, 'utf-8'))
         jsn = msgspec.json.decode(byte_data, type=BehaviorReg)
         while counter < len(jsn.behavior.enhanced):
             if jsn.behavior.enhanced[counter]['object'] == "registry":
                 if feature in jsn.behavior.enhanced[counter]["data"]["regkey"]:
                     return 1
             counter += 1
    return 0

reg_names = ["Cryptography", "IEData", "Tcpip", "Dnscache", "DockingState", "CustomLocale", "SafeBoot", "Nls\\Sorting", "SystemInformation", "Persistence" ]
         
def registry_calls(dictionary):
    reg_vector = []
    for name in reg_names:
        reg_vector.append(registry_finder(dictionary, name))
    return reg_vector
#registry_full = registry_calls(r"C:/Users/T pro/Desktop/ffe7f8921b0d42d3014c972c33460ec0f6565e4ce0bb8bd04df4a6d635c7903d.json")
    
    
    
    
    
    
    
    


# ----------------------- DEFINITION API CALLS--------------------
class Processes(Struct):
    processes: list

class Behavior(Struct):
    behavior: Processes
    

api_names = ["NtDelayExecution", "NtCreateFile", "NtFreeVirtualMemory", "HttpOpenRequestA", "NtOpenFile", "Socket", "CryptDecodeObjectEx", "OpenSCManagerA", "CryptGenKey", "CryptAcquireContextA", "NtAllocateVirtualMemory", "bind", "closesocket", "NtCreateMutant", "DeviceIoControl", "GetSystemTimeAsFileTime", "HttpSendRequestA", "NtMapViewOfSection", "NtOpenMutant", "NtProtectVirtualMemory", "NtWriteFile", "CreateToolhelp32Snapshot", "CreateRemoteThread", "NtDuplicateObject", "NtQueryInformationFile", "InternetReadFile", "CryptCreateHash", "CryptHashData", "CheckCursorPos", "CryptExportKey"]


    
    
def api_calls(dictionary, feature):
    counter = 0
    with open(dictionary, "r") as f:
         data = f.read()
         byte_data = (bytes(data, 'utf-8'))
         calls = msgspec.json.decode(byte_data, type=Behavior)
         for call in calls.behavior.processes:
             for c in call['calls']:
                 if c['api'] == feature:
                     counter += 1
    return counter

def api_vector(dictionary):
    api_vec = []
    for name in api_names:
        api_vec.append(api_calls(dictionary, name))
    return api_vec

# ----------------------------------------------------------------------------#



def vector_file(dictionary):
    path = dictionary.path
    file_vector = []
    file_vector =  [path[61:-5]] + (network_calls(dictionary)) + (static_calls(dictionary)) + (registry_calls(dictionary)) + (api_vector(dictionary))
    file_vector.append(behav_calls(dictionary))
    file_vector.append(dropped_calls(dictionary))
    return file_vector


def input_vector(folder, start, end):
    full_input = []

    counter = 0
    files = os.scandir(folder)
    for file in files:
        if (counter >= start) and (counter <= end):
            full_input.append(vector_file(file))
        counter = counter + 1

    return full_input

# folder = r"/mnt/data/jsonlearning/Avast_cuckoo_full/public_full_reports"
folder = r"reports"
folder2 = r"results"

all_input =  input_vector(folder, int(sys.argv[1]), int(sys.argv[2]))

np.savetxt(f'input_{sys.argv[1]}_{sys.argv[2]}.csv', 
           all_input,
           delimiter =", ", 
           fmt ='% s')


# =============================================================================
# print("It took: {0} s".format(end - start))
# print("The whole bunch would take: {0} hours".format((end - start) * (50000/len(all_input)) / 3600))
# =============================================================================
