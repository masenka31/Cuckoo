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
# ------------------------------ N E T W O R K ------------------------------
print("----------------TRY----------------")

class Network(Struct):
    network: dict

# Test if the results are the same - the structures mightno tb e needed?
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

# ------------------------------ S T A T I C ------------------------------
class PE(Struct):
    pe: dict

class Static(Struct):
    static: PE

def stat_pe(dictionary, feature):
    with open(dictionary, "r") as f:
         data = f.read()
         byte_data = (bytes(data, 'utf-8'))
         try: 
             jsn = msgspec.json.decode(byte_data, type=Static)
             for key in jsn.static:
                 if key == feature:
                     return len(jsn.static[feature])
         except:
            return 0

static_names = ["pe_sections", "pe_exports", "pe_resources"]

def static_calls(dictionary):
    static_vector = []
    for name in static_names:
        static_vector.append(stat_pe(dictionary, name))
    return static_vector

# static_calls("reports/report.json")

# ------------------------------ B E H A V I O U R ------------------------------

class Processes(Struct):
    processes: list

class Behavior(Struct):
    behavior: Processes


def behav_calls(dictionary):
    with open(dictionary, "r") as f:
        data = f.read()
        byte_data = (bytes(data, 'utf-8'))
        try: 
            jsn = msgspec.json.decode(byte_data, type=Behavior)
            return len(jsn.behavior.processes)
        except:
            print('xxx')
            return 0

# behav_calls("reports/report2.json")

# ------------------------------ D R O P P E D ------------------------------

class Dropped(Struct):
    dropped: list


def dropped_calls(dictionary):
    with open(dictionary, "r") as f:
        data = f.read()
        byte_data = (bytes(data, 'utf-8'))
        try: 
            jsn = msgspec.json.decode(byte_data, type=Dropped)
            return len(jsn.dropped)
        except:
            return 0

# dropped_calls("reports/report4.json")

# ------------------------------ R E G I S T R Y  K E Y S ------------------------------

reg_keys = ["regkey_deleted", "regkey_opened", "regkey_read", "regkey_written"]
reg_names = ["Cryptography", "IEData", "Tcpip", "Dnscache", "DockingState", "CustomLocale", "SafeBoot", "Nls\\Sorting", "SystemInformation", "Persistence" ]

class Summary(Struct):
    summary: dict

class SummaryBehavior(Struct):
    behavior: Summary

def registry_finder(dictionary, feature):
    with open(dictionary, "r") as f:
        data = f.read()
        byte_data = (bytes(data, 'utf-8'))
        try:
            jsn = msgspec.json.decode(byte_data, type=SummaryBehavior)
        except:
            return 0
        
        jsn = msgspec.json.decode(byte_data, type=SummaryBehavior)
        for regkey in reg_keys:
            if regkey in jsn.behavior.summary.keys():
                for x in jsn.behavior.summary[regkey]:
                    if feature in x:
                        return 1
    
    return 0

def registry_calls(dictionary):
    reg_vector = []
    for name in reg_names:
        reg_vector.append(registry_finder(dictionary, name))
    return reg_vector

# registry_calls("reports/report.json")

# dictionary = "reports/report2.json"
# with open(dictionary, "r") as f:
#     data = f.read()
#     byte_data = bytes(data, 'utf-8')
    
#     try:
#         jsn = msgspec.json.decode(byte_data, type=SummaryBehavior)
#     except:
#         print(0)
    # behavior = msgspec.json.decode(byte_data, type=SummaryBehavior)

# ------------------------------ DEFINITION API CALLS ------------------------------

api_names = ["NtDelayExecution", "NtCreateFile", "NtFreeVirtualMemory", "HttpOpenRequestA", "NtOpenFile", "Socket", "CryptDecodeObjectEx", "OpenSCManagerA", "CryptGenKey", "CryptAcquireContextA", "NtAllocateVirtualMemory", "bind", "closesocket", "NtCreateMutant", "DeviceIoControl", "GetSystemTimeAsFileTime", "HttpSendRequestA", "NtMapViewOfSection", "NtOpenMutant", "NtProtectVirtualMemory", "NtWriteFile", "CreateToolhelp32Snapshot", "CreateRemoteThread", "NtDuplicateObject", "NtQueryInformationFile", "InternetReadFile", "CryptCreateHash", "CryptHashData", "CheckCursorPos", "CryptExportKey"]

def api_calls(dictionary, feature):
    with open(dictionary, "r") as f:
        counter = 0
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

# dictionary = "reports/report.json"
# api_vector(dictionary)

# ------------------------------ feature calculation ------------------------------

def vector_file(dictionary):
    file_vector = []
    file_vector =  [dictionary[-69:-5]] + (network_calls(dictionary)) + (static_calls(dictionary)) + (registry_calls(dictionary)) + (api_vector(dictionary))
    file_vector.append(behav_calls(dictionary))
    file_vector.append(dropped_calls(dictionary))
    return file_vector

import os

# def list_report_files(root_path):
#     report_files = []
#     for folder_name in ['malicious', 'benign']:
#         folder_path = os.path.join(root_path, folder_name)
#         for i in range(10):
#             subfolder_path = os.path.join(folder_path, str(i))
#             if not os.path.exists(subfolder_path):
#                 continue
#             for j in range(1000):
#                 file_path = os.path.join(subfolder_path, str(j), 'report.json')
#                 if os.path.isfile(file_path):
#                     report_files.append(file_path)
#     return report_files

def list_report_files(root_path):
    report_files = []
    folders = os.listdir(root_path)
    for folder in folders:
        files = os.listdir(os.path.join(root_path, folder))
        for file in files:
            if file.endswith('json'):
                report_files.append(os.path.join(root_path, folder, file))
                
    return report_files



root_path = "/mnt/data/jsonlearning/datasets/garcia/reports"

# files = list_report_files(root_path)
# for file_path in files:
#     print(file_path)

def input_vector(root_path, start, end):
    files = list_report_files(root_path)
    chosen_files = files[start:end]
    full_input = []
    
    for file in chosen_files:
        try:
            full_input.append(vector_file(file))
        except:
            pass
    
    return full_input

# folder = r"/mnt/data/jsonlearning/Avast_cuckoo_full/public_full_reports"
# folder = r"reports"

all_input =  input_vector(root_path, int(sys.argv[1]), int(sys.argv[2]))

np.savetxt(f'files/input_{sys.argv[1]}_{sys.argv[2]}.csv', 
           all_input,
           delimiter =", ", 
           fmt ='% s')
